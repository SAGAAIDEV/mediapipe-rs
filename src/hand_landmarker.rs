use std::num::NonZeroU32;
use std::os::raw::c_void;
use std::ptr;

use crate::error::{Error, Result};
use crate::face_detector::status_only_error;
use crate::image::{Image, ImageRef};
use crate::loader::lib;
use crate::stream::{AsyncTask, Slot, Stream};
use crate::sys::{self, Abi, ModelHold};
use crate::types::{
    Category, Confidence, Delegate, IouThreshold, ModelSource, NormalizedLandmark, Rotation,
    Timestamp, WorldLandmark,
};

/// One hand found in a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hand {
    /// Which hand the model thinks this is, best first — `"Left"` or `"Right"`
    /// in `category_name`.
    ///
    /// MediaPipe labels hands **assuming the image is mirrored**, as a selfie
    /// camera's preview is. Fed an unmirrored frame, the label names the
    /// opposite hand of the person in it.
    pub handedness: Vec<Category>,
    /// 21 landmarks in image space: the wrist, then four per finger from the
    /// thumb outwards, knuckle to tip.
    pub landmarks: Vec<NormalizedLandmark>,
    /// The same 21 points in metres, around the hand's own centre. What hand
    /// *orientation* should be computed from: unlike the image-space `z`, all
    /// three axes share a unit.
    pub world_landmarks: Vec<WorldLandmark>,
}

/// Every hand found in one frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandLandmarkerResult {
    pub hands: Vec<Hand>,
}

impl HandLandmarkerResult {
    /// # Safety
    /// `raw` must be a live `MpHandLandmarkerResult`.
    unsafe fn from_raw(raw: &sys::MpHandLandmarkerResult) -> Self {
        // The three arrays are parallel, one entry per hand. They are filled
        // together in `hand_landmarker_result_converter.cc`, but nothing in the
        // C struct promises equal counts, so the landmark array decides how many
        // hands there are and the other two contribute an entry where they have
        // one rather than being indexed past their own count.
        if raw.hand_landmarks.is_null() {
            return HandLandmarkerResult::default();
        }
        let hands = (0..raw.hand_landmarks_count as usize)
            .map(|h| {
                // SAFETY: the index is below the count MediaPipe reported for this array.
                let set = unsafe { &*raw.hand_landmarks.add(h) };
                let landmarks = if set.landmarks.is_null() {
                    Vec::new()
                } else {
                    (0..set.landmarks_count as usize)
                        // SAFETY: the index is below the count MediaPipe reported for this array.
                        .map(|i| unsafe { NormalizedLandmark::from_raw(&*set.landmarks.add(i)) })
                        .collect()
                };

                let handedness = if raw.handedness.is_null() || h >= raw.handedness_count as usize {
                    Vec::new()
                } else {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.handedness.add(h) };
                    // SAFETY: a field of the live result this function's contract covers.
                    unsafe { Category::vec_from_raw(set.categories, set.categories_count) }
                };

                let world_landmarks = if raw.hand_world_landmarks.is_null()
                    || h >= raw.hand_world_landmarks_count as usize
                {
                    Vec::new()
                } else {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.hand_world_landmarks.add(h) };
                    if set.landmarks.is_null() {
                        Vec::new()
                    } else {
                        (0..set.landmarks_count as usize)
                            // SAFETY: the index is below the count MediaPipe reported for this array.
                            .map(|i| unsafe { WorldLandmark::from_raw(&*set.landmarks.add(i)) })
                            .collect()
                    }
                };

                Hand {
                    handedness,
                    landmarks,
                    world_landmarks,
                }
            })
            .collect();
        HandLandmarkerResult { hands }
    }
}

#[derive(Debug, Clone)]
pub struct HandLandmarkerBuilder {
    source: ModelSource,
    delegate: Delegate,
    num_hands: NonZeroU32,
    min_hand_detection_confidence: Confidence,
    min_hand_presence_confidence: Confidence,
    min_tracking_confidence: IouThreshold,
}

impl HandLandmarkerBuilder {
    pub fn delegate(mut self, delegate: Delegate) -> Self {
        self.delegate = delegate;
        self
    }

    /// Maximum number of hands to track.
    pub fn num_hands(mut self, n: NonZeroU32) -> Self {
        self.num_hands = n;
        self
    }

    /// Minimum score for the palm detector, the stage that finds hands the
    /// tracker is not already following.
    pub fn min_hand_detection_confidence(mut self, v: Confidence) -> Self {
        self.min_hand_detection_confidence = v;
        self
    }

    /// Minimum score that a hand is still present in the region being tracked.
    /// Below it the hand is dropped and the palm detector looks again.
    pub fn min_hand_presence_confidence(mut self, v: Confidence) -> Self {
        self.min_hand_presence_confidence = v;
        self
    }

    /// How much a newly detected hand box must overlap a tracked one to count
    /// as the same hand.
    ///
    /// Upstream calls this `min_tracking_confidence`, but as with the face
    /// landmarker the name is wrong: `hand_landmarker_graph.cc` passes it to
    /// `HandAssociationCalculator` as `min_similarity_threshold`, which is
    /// compared against `CalculateIou`. It is an overlap ratio.
    ///
    /// It must also be **above zero**: that calculator `ABSL_CHECK`s it in
    /// `Open`, which aborts the process rather than returning an error, so
    /// `build_*` refuses `0.0` before MediaPipe sees it.
    pub fn min_tracking_confidence(mut self, v: IouThreshold) -> Self {
        self.min_tracking_confidence = v;
        self
    }

    pub fn build(self) -> Result<HandLandmarker> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_IMAGE, None)
    }

    pub fn build_for_video(self) -> Result<HandLandmarker> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_VIDEO, None)
    }

    pub fn build_stream<F>(self, callback: F) -> Result<HandLandmarkerStream>
    where
        F: FnMut(Result<HandLandmarkerResult>, ImageRef<'_>, Timestamp) + Send + 'static,
    {
        let mut callback = callback;
        let slot = Slot::claim(Box::new(move |status, result, image, ts| {
            // Copy out before returning: MediaPipe frees the result immediately
            // afterwards, and `image` is a stack MpImage in its dispatch lambda.
            let converted = if status != sys::MpStatus::kMpOk || result.is_null() {
                Err(status_only_error(status))
            } else {
                // SAFETY: status is kMpOk and `result` is non-null, so MediaPipe
                // handed us a live result of this task's type.
                Ok(unsafe {
                    HandLandmarkerResult::from_raw(&*result.cast::<sys::MpHandLandmarkerResult>())
                })
            };
            callback(
                converted,
                ImageRef::from_raw(image),
                Timestamp::from_millis(ts),
            );
        }))?;

        let raw_cb = slot.raw_callback();
        let landmarker = self.build_mode(
            sys::MpRunningMode::MP_RUNNING_MODE_LIVE_STREAM,
            Some(raw_cb),
        )?;
        Ok(Stream::new(landmarker, slot))
    }

    fn build_mode(
        self,
        running_mode: sys::MpRunningMode,
        callback: Option<unsafe extern "C" fn(sys::MpStatus, *const c_void, sys::MpImagePtr, i64)>,
    ) -> Result<HandLandmarker> {
        let min_hand_detection_confidence = self.min_hand_detection_confidence.get();
        let min_hand_presence_confidence = self.min_hand_presence_confidence.get();
        let min_tracking_confidence = self.min_tracking_confidence.get();
        // See `min_tracking_confidence`: zero is in range for an IoU threshold
        // but aborts this task's graph.
        if min_tracking_confidence <= 0.0 {
            return Err(Error::OutOfRange {
                quantity: "hand-tracking IoU threshold (HandAssociationCalculator aborts on 0)",
                value: min_tracking_confidence,
                range: f32::MIN_POSITIVE..=1.0,
            });
        }

        let lib = lib()?;
        let hold = ModelHold::new(&self.source)?;
        let (buf, buf_count, path) = hold.parts();
        let delegate = self.delegate.to_raw();
        // A C `int`; clamp rather than wrap, as the face landmarker does.
        let num_hands = i32::try_from(self.num_hands.get()).unwrap_or(i32::MAX);
        let cb: sys::MpHandLandmarkerOptions_result_callback_fn =
            // SAFETY: the two fn types differ only in the pointee of the result pointer, which does not change the C ABI.
            callback.map(|f| unsafe { std::mem::transmute(f) });

        let mut ptr: sys::MpHandLandmarkerPtr = ptr::null_mut();
        let mut err = ptr::null_mut();

        // SAFETY: `hold` keeps the model alive across the call; the options
        // layout is chosen from the probed ABI (see sys::compat).
        let status = unsafe {
            match lib.abi {
                Abi::Renamed => {
                    let mut opts = sys::MpHandLandmarkerOptions {
                        base_options: sys::MpBaseOptions {
                            model_asset_buffer: buf,
                            model_asset_buffer_count: buf_count,
                            model_asset_path: path,
                            file_descriptor: 0,
                            delegate,
                            host_environment: sys::MpHostEnvironment::MP_HOST_ENVIRONMENT_UNKNOWN,
                            host_system: sys::MpHostSystem::MP_HOST_SYSTEM_UNKNOWN,
                            host_version: ptr::null(),
                            ca_bundle_path: ptr::null(),
                            app_id: ptr::null(),
                            app_version: ptr::null(),
                        },
                        running_mode,
                        num_hands,
                        min_hand_detection_confidence,
                        min_hand_presence_confidence,
                        min_tracking_confidence,
                        result_callback: cb,
                    };
                    lib.raw
                        .MpHandLandmarkerCreate(&mut opts, &mut ptr, &mut err)
                }
                Abi::V0_10_35 => {
                    let mut opts = sys::compat::MpHandLandmarkerOptionsV35 {
                        base_options: sys::compat::MpBaseOptionsV35::new(
                            buf, buf_count, path, delegate,
                        ),
                        running_mode,
                        num_hands,
                        min_hand_detection_confidence,
                        min_hand_presence_confidence,
                        min_tracking_confidence,
                        result_callback: cb,
                    };
                    lib.raw.MpHandLandmarkerCreate(
                        (&raw mut opts).cast::<sys::MpHandLandmarkerOptions>(),
                        &mut ptr,
                        &mut err,
                    )
                }
            }
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(HandLandmarker { ptr, closed: false })
    }
}

/// 21 landmarks per hand, with handedness, in image space and in metres.
pub struct HandLandmarker {
    ptr: sys::MpHandLandmarkerPtr,
    closed: bool,
}

// SAFETY: as FaceDetector — movable between threads, never shared.
unsafe impl Send for HandLandmarker {}

impl HandLandmarker {
    pub fn builder(source: ModelSource) -> HandLandmarkerBuilder {
        HandLandmarkerBuilder {
            source,
            delegate: Delegate::default(),
            num_hands: NonZeroU32::new(1).expect("1 should be non-zero"),
            // MediaPipe's own defaults.
            min_hand_detection_confidence: Confidence::HALF,
            min_hand_presence_confidence: Confidence::HALF,
            min_tracking_confidence: IouThreshold::HALF,
        }
    }

    pub fn detect(&mut self, image: &Image) -> Result<HandLandmarkerResult> {
        self.detect_inner(image, None, None)
    }

    /// Applies a clockwise rotation before inference.
    ///
    /// There is deliberately no region-of-interest variant: `hand_landmarker.cc`
    /// builds every mode with `roi_allowed=false`.
    pub fn detect_rotated(
        &mut self,
        image: &Image,
        rotation: Rotation,
    ) -> Result<HandLandmarkerResult> {
        let raw = rotation.to_raw();
        self.detect_inner(image, Some(&raw), None)
    }

    /// Video mode. Timestamps must increase between calls.
    ///
    /// The mode worth using on a camera: once a hand is found, later frames
    /// track it from the previous landmarks instead of running the palm
    /// detector again.
    pub fn detect_for_video(
        &mut self,
        image: &Image,
        timestamp: Timestamp,
    ) -> Result<HandLandmarkerResult> {
        self.detect_inner(image, None, Some(timestamp.as_millis()))
    }

    fn detect_inner(
        &mut self,
        image: &Image,
        processing: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: Option<i64>,
    ) -> Result<HandLandmarkerResult> {
        let lib = lib()?;
        let opts = processing.map_or(ptr::null(), |p| p as *const _);
        // SAFETY: the struct is only pointers and counts, so all-zero is a valid empty result; MediaPipe overwrites it.
        let mut result: sys::MpHandLandmarkerResult = unsafe { std::mem::zeroed() };
        let mut err = ptr::null_mut();

        // SAFETY: live handle and image; `result` goes straight back to
        // MpHandLandmarkerCloseResult.
        let status = unsafe {
            match timestamp_ms {
                None => lib.raw.MpHandLandmarkerDetectImage(
                    self.ptr,
                    image.ptr,
                    opts,
                    &mut result,
                    &mut err,
                ),
                Some(ts) => lib.raw.MpHandLandmarkerDetectForVideo(
                    self.ptr,
                    image.ptr,
                    opts,
                    ts,
                    &mut result,
                    &mut err,
                ),
            }
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }

        // Convert first, free second.
        // SAFETY: `result` was filled by the call above and is still live.
        let converted = unsafe { HandLandmarkerResult::from_raw(&result) };
        // SAFETY: `result` was filled by the Detect call above, everything has been copied out, and it is freed once.
        unsafe { lib.raw.MpHandLandmarkerCloseResult(&mut result) };
        Ok(converted)
    }

    fn close_once(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: runs exactly once; joins any worker thread.
        let status = unsafe { lib.raw.MpHandLandmarkerClose(self.ptr, &mut err) };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }
}

impl std::fmt::Debug for HandLandmarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandLandmarker")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Drop for HandLandmarker {
    fn drop(&mut self) {
        let _ = self.close_once();
    }
}

impl AsyncTask for HandLandmarker {
    fn send_raw(
        &mut self,
        image: &Image,
        opts: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: i64,
    ) -> Result<()> {
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: as detect_inner.
        let status = unsafe {
            lib.raw.MpHandLandmarkerDetectAsync(
                self.ptr,
                image.ptr,
                opts.map_or(ptr::null(), |p| p as *const _),
                timestamp_ms,
                &mut err,
            )
        };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.close_once()
    }
}

/// A [`HandLandmarker`] running in live-stream mode.
pub type HandLandmarkerStream = Stream<HandLandmarker>;

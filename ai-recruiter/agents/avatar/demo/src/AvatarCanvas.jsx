/**
 * AvatarCanvas — Full-body 3D avatar with production lip sync and
 * natural conversational gestures.
 *
 * FIXES (v4):
 *  ROOT CAUSE: RPM T-pose has arms at 90° from vertical (horizontal).
 *  Bringing them DOWN requires a large rotation:
 *    RightArm z → must be POSITIVE +1.35 to adduct arm downward
 *    LeftArm  z → must be NEGATIVE -1.35
 *  Previous attempts:
 *    v2: z: -1.55/+1.55 → WRONG SIGN, arms went UP toward head
 *    v3: z: +0.18/-0.18 → RIGHT SIGN but only ~10° down, still near T-pose
 *    v4: z: +1.35/-1.35 → RIGHT SIGN, correct magnitude (~77° = natural hang)
 *
 *  Also added a console.log of actual bone rotations at load so you can
 *  verify/tune if the rig differs from standard RPM convention.
 */

import { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";

const AVATAR_URL = "/avatar/model.glb";
const IDLE_BLINK_INTERVAL = 3200;
const IDLE_BLINK_DURATION = 160;

// ─── Natural rest pose ────────────────────────────────────────────────────────
// RPM/Mixamo rig: T-pose arms are horizontal (z≈0 for upper arm bones).
//   RightArm +z → arm rotates DOWN (adduction toward body)
//   RightArm -z → arm rotates UP  (abduction away from body)
// Natural hang ≈ 77° below horizontal → z: +1.35 for right, -1.35 for left.
const NATURAL_POSE = {
  RightShoulder: { x:  0.00, y:  0.00, z: -0.06 },
  LeftShoulder:  { x:  0.00, y:  0.00, z:  0.06 },

  RightArm:      { x:  0.00, y:  0.00, z:  1.50 },  // only z rotation = straight down
  LeftArm:       { x:  0.00, y:  0.00, z: -1.50 },  // only z rotation = straight down

  RightForeArm:  { x:  0.00, y: -0.30, z:  0.00 },  // slight inward bend only
  LeftForeArm:   { x:  0.00, y:  0.30, z:  0.00 },

  RightHand:     { x:  0.00, y:  0.00, z:  0.00 },
  LeftHand:      { x:  0.00, y:  0.00, z:  0.00 },
};

// ─── Conversational gesture library ──────────────────────────────────────────
// Offsets are ADDED on top of NATURAL_POSE. Max ≈ 0.22 rad (≈ 13°).
const GESTURE_LIBRARY = [
  // 1. Right forearm slight lift — "making a point"
  {
    duration: 2.2,
    bones: [
      { t: 0.0, RightForeArm: { x:  0.00, y:  0.00, z: 0 }, RightHand: { x:  0.00, y:  0.00, z:  0.00 } },
      { t: 0.4, RightForeArm: { x: -0.16, y:  0.06, z: 0 }, RightHand: { x: -0.07, y: -0.04, z:  0.04 } },
      { t: 1.1, RightForeArm: { x: -0.13, y:  0.04, z: 0 }, RightHand: { x: -0.05, y: -0.03, z:  0.03 } },
      { t: 1.8, RightForeArm: { x: -0.04, y:  0.01, z: 0 }, RightHand: { x: -0.01, y:  0.00, z:  0.01 } },
      { t: 2.2, RightForeArm: { x:  0.00, y:  0.00, z: 0 }, RightHand: { x:  0.00, y:  0.00, z:  0.00 } },
    ],
  },
  // 2. Both forearms open — welcoming gesture
  {
    duration: 3.0,
    bones: [
      { t: 0.0, LeftForeArm: { x: 0, y: 0, z: 0 }, RightForeArm: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 }, RightHand: { x: 0, y: 0, z: 0 } },
      { t: 0.6, LeftForeArm: { x: -0.14, y: -0.06, z: 0 }, RightForeArm: { x: -0.14, y: 0.06, z: 0 }, LeftHand: { x: -0.05, y: 0.06, z: 0 }, RightHand: { x: -0.05, y: -0.06, z: 0 } },
      { t: 1.5, LeftForeArm: { x: -0.11, y: -0.04, z: 0 }, RightForeArm: { x: -0.11, y: 0.04, z: 0 }, LeftHand: { x: -0.03, y: 0.04, z: 0 }, RightHand: { x: -0.03, y: -0.04, z: 0 } },
      { t: 2.4, LeftForeArm: { x: -0.03, y: -0.01, z: 0 }, RightForeArm: { x: -0.03, y: 0.01, z: 0 }, LeftHand: { x: -0.01, y: 0.01, z: 0 }, RightHand: { x: -0.01, y: -0.01, z: 0 } },
      { t: 3.0, LeftForeArm: { x: 0, y: 0, z: 0 }, RightForeArm: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 }, RightHand: { x: 0, y: 0, z: 0 } },
    ],
  },
  // 3. Left forearm lift — emphasising
  {
    duration: 2.0,
    bones: [
      { t: 0.0, LeftForeArm: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 } },
      { t: 0.4, LeftForeArm: { x: -0.18, y: -0.07, z: 0 }, LeftHand: { x: 0.06, y: -0.04, z: -0.04 } },
      { t: 1.0, LeftForeArm: { x: -0.14, y: -0.05, z: 0 }, LeftHand: { x: 0.04, y: -0.02, z: -0.03 } },
      { t: 1.6, LeftForeArm: { x: -0.04, y: -0.01, z: 0 }, LeftHand: { x: 0.01, y: 0, z: -0.01 } },
      { t: 2.0, LeftForeArm: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 } },
    ],
  },
  // 4. Right wrist nod — affirmation
  {
    duration: 1.4,
    bones: [
      { t: 0.0, RightHand: { x: 0, y: 0, z: 0 }, RightForeArm: { x: 0, y: 0, z: 0 } },
      { t: 0.2, RightHand: { x: -0.09, y: 0, z: 0.03 }, RightForeArm: { x: -0.06, y: 0, z: 0 } },
      { t: 0.5, RightHand: { x: -0.06, y: 0.01, z: 0.02 }, RightForeArm: { x: -0.04, y: 0, z: 0 } },
      { t: 0.9, RightHand: { x: -0.02, y: 0, z: 0.01 }, RightForeArm: { x: -0.01, y: 0, z: 0 } },
      { t: 1.4, RightHand: { x: 0, y: 0, z: 0 }, RightForeArm: { x: 0, y: 0, z: 0 } },
    ],
  },
  // 5. Left wrist flick — quick emphasis
  {
    duration: 1.2,
    bones: [
      { t: 0.0, LeftHand: { x: 0, y: 0, z: 0 }, LeftForeArm: { x: 0, y: 0, z: 0 } },
      { t: 0.2, LeftHand: { x: 0.08, y: -0.03, z: -0.04 }, LeftForeArm: { x: -0.07, y: 0, z: 0 } },
      { t: 0.5, LeftHand: { x: 0.05, y: -0.02, z: -0.02 }, LeftForeArm: { x: -0.04, y: 0, z: 0 } },
      { t: 0.9, LeftHand: { x: 0.02, y: 0, z: -0.01 }, LeftForeArm: { x: -0.01, y: 0, z: 0 } },
      { t: 1.2, LeftHand: { x: 0, y: 0, z: 0 }, LeftForeArm: { x: 0, y: 0, z: 0 } },
    ],
  },
  // 6. Gentle right arm sway — idle weight shift
  {
    duration: 3.5,
    bones: [
      { t: 0.0, RightArm: { x: 0, y: 0, z: 0 }, RightHand: { x: 0, y: 0, z: 0 } },
      { t: 0.9, RightArm: { x: -0.04, y: 0.03, z: 0 }, RightHand: { x: -0.04, y: -0.03, z: 0.02 } },
      { t: 2.0, RightArm: { x: -0.02, y: 0.02, z: 0 }, RightHand: { x: -0.02, y: -0.02, z: 0.01 } },
      { t: 3.0, RightArm: { x: -0.01, y: 0.01, z: 0 }, RightHand: { x: -0.01, y: 0, z: 0 } },
      { t: 3.5, RightArm: { x: 0, y: 0, z: 0 }, RightHand: { x: 0, y: 0, z: 0 } },
    ],
  },
  // 7. Both hands pulse — listing / counting
  {
    duration: 1.8,
    bones: [
      { t: 0.0, RightHand: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 } },
      { t: 0.3, RightHand: { x: -0.08, y: -0.03, z: 0.02 }, LeftHand: { x: -0.08, y: 0.03, z: -0.02 } },
      { t: 0.7, RightHand: { x: -0.06, y: -0.02, z: 0.02 }, LeftHand: { x: -0.06, y: 0.02, z: -0.02 } },
      { t: 1.3, RightHand: { x: -0.02, y: -0.01, z: 0.01 }, LeftHand: { x: -0.02, y: 0.01, z: -0.01 } },
      { t: 1.8, RightHand: { x: 0, y: 0, z: 0 }, LeftHand: { x: 0, y: 0, z: 0 } },
    ],
  },
];

function interpolateGesture(gesture, elapsed) {
  const kf = gesture.bones;
  if (elapsed <= kf[0].t) return kf[0];
  if (elapsed >= kf[kf.length - 1].t) return kf[kf.length - 1];
  let a = kf[0], b = kf[1];
  for (let i = 0; i < kf.length - 1; i++) {
    if (elapsed >= kf[i].t && elapsed < kf[i + 1].t) { a = kf[i]; b = kf[i + 1]; break; }
  }
  const p = (elapsed - a.t) / (b.t - a.t);
  const s = p * p * (3 - 2 * p);
  const result = {};
  const allBones = new Set([...Object.keys(a), ...Object.keys(b)]);
  allBones.delete("t");
  for (const bone of allBones) {
    const av = a[bone] || { x: 0, y: 0, z: 0 };
    const bv = b[bone] || { x: 0, y: 0, z: 0 };
    result[bone] = { x: av.x + (bv.x - av.x) * s, y: av.y + (bv.y - av.y) * s, z: av.z + (bv.z - av.z) * s };
  }
  return result;
}

const FINGER_BONES = [
  "RightHandThumb1","RightHandThumb2","RightHandThumb3",
  "RightHandIndex1","RightHandIndex2","RightHandIndex3",
  "RightHandMiddle1","RightHandMiddle2","RightHandMiddle3",
  "RightHandRing1","RightHandRing2","RightHandRing3",
  "RightHandPinky1","RightHandPinky2","RightHandPinky3",
  "LeftHandThumb1","LeftHandThumb2","LeftHandThumb3",
  "LeftHandIndex1","LeftHandIndex2","LeftHandIndex3",
  "LeftHandMiddle1","LeftHandMiddle2","LeftHandMiddle3",
  "LeftHandRing1","LeftHandRing2","LeftHandRing3",
  "LeftHandPinky1","LeftHandPinky2","LeftHandPinky3",
];

const ANIMATED_BONES = [
  "Head","Neck","Spine1","Spine2",
  "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
  "RightShoulder","RightArm","RightForeArm","RightHand",
  ...FINGER_BONES,
];

const FINGER_REST_CURL = {
  RightHandThumb1:{x:0.15,y:0,z:0.20},RightHandThumb2:{x:0.10,y:0,z:0},RightHandThumb3:{x:0.08,y:0,z:0},
  RightHandIndex1:{x:0.22,y:0,z:0},RightHandIndex2:{x:0.26,y:0,z:0},RightHandIndex3:{x:0.18,y:0,z:0},
  RightHandMiddle1:{x:0.24,y:0,z:0},RightHandMiddle2:{x:0.28,y:0,z:0},RightHandMiddle3:{x:0.20,y:0,z:0},
  RightHandRing1:{x:0.26,y:0,z:0},RightHandRing2:{x:0.30,y:0,z:0},RightHandRing3:{x:0.22,y:0,z:0},
  RightHandPinky1:{x:0.28,y:0,z:0},RightHandPinky2:{x:0.32,y:0,z:0},RightHandPinky3:{x:0.24,y:0,z:0},
  LeftHandThumb1:{x:0.15,y:0,z:-0.20},LeftHandThumb2:{x:0.10,y:0,z:0},LeftHandThumb3:{x:0.08,y:0,z:0},
  LeftHandIndex1:{x:0.22,y:0,z:0},LeftHandIndex2:{x:0.26,y:0,z:0},LeftHandIndex3:{x:0.18,y:0,z:0},
  LeftHandMiddle1:{x:0.24,y:0,z:0},LeftHandMiddle2:{x:0.28,y:0,z:0},LeftHandMiddle3:{x:0.20,y:0,z:0},
  LeftHandRing1:{x:0.26,y:0,z:0},LeftHandRing2:{x:0.30,y:0,z:0},LeftHandRing3:{x:0.22,y:0,z:0},
  LeftHandPinky1:{x:0.28,y:0,z:0},LeftHandPinky2:{x:0.32,y:0,z:0},LeftHandPinky3:{x:0.24,y:0,z:0},
};

function AvatarModel({ lipSyncRef, isSpeaking, isThinking, controlsRef }) {
  const { scene }  = useGLTF(AVATAR_URL);
  const { camera } = useThree();

  const meshesRef          = useRef([]);
  const bonesRef           = useRef({});
  const origRotRef         = useRef({});
  const blinkTimerRef      = useRef(0);
  const lastBlinkRef       = useRef(0);
  const idleTimeRef        = useRef(0);
  const cameraReady        = useRef(false);
  const initDone           = useRef(false);
  const gestureRef         = useRef(null);
  const gestureTimeRef     = useRef(0);
  const gestureCooldownRef = useRef(1.8);
  const speechAmpRef       = useRef(0);

  useEffect(() => {
    const morphMeshes = [];
    const boneMap     = {};
    scene.traverse((child) => {
      if (child.isMesh && child.morphTargetDictionary) morphMeshes.push(child);
      if (child.isBone) boneMap[child.name] = child;
    });
    meshesRef.current = morphMeshes;
    bonesRef.current  = boneMap;

    if (!initDone.current) {
      initDone.current = true;
      const orig = {};
      for (const name of ANIMATED_BONES) {
        const bone = boneMap[name];
        if (bone) orig[name] = { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z };
      }
      origRotRef.current = orig;

      // Debug: verify actual rig rotations
      ["RightArm","LeftArm","RightForeArm","LeftForeArm"].forEach(n => {
        const b = boneMap[n];
        if (b) console.log(`[Avatar] ${n} orig:`, b.rotation.x.toFixed(3), b.rotation.y.toFixed(3), b.rotation.z.toFixed(3));
      });

      // Apply NATURAL_POSE immediately so arms start in resting position
      // (avoids the slow lerp-from-T-pose on first load)
      for (const [boneName, offset] of Object.entries(NATURAL_POSE)) {
        const bone = boneMap[boneName];
        const o = orig[boneName];
        if (bone && o) {
          bone.rotation.x = o.x + offset.x;
          bone.rotation.y = o.y + offset.y;
          bone.rotation.z = o.z + offset.z;
        }
      }
      console.log("[Avatar] Applied NATURAL_POSE directly on init");
    }

    if (!cameraReady.current) {
      cameraReady.current = true;
      const box = new THREE.Box3().setFromObject(scene);
      const size = new THREE.Vector3(); const center = new THREE.Vector3();
      box.getSize(size); box.getCenter(center);
      const targetY = box.min.y + size.y * 0.62;
      const fov = 28;
      const dist = (size.y * 1.15 / 2) / Math.tan(THREE.MathUtils.degToRad(fov / 2));
      camera.position.set(center.x, targetY, center.z + Math.max(dist, 3.2));
      camera.fov = fov; camera.near = 0.01; camera.far = 60;
      camera.updateProjectionMatrix();
      if (controlsRef?.current) {
        controlsRef.current.target.set(center.x, targetY, center.z);
        controlsRef.current.update();
      }
    }
  }, [scene, camera, controlsRef]);

  const setBoneWithPose = (boneName, gestureOffset, lerpFactor) => {
    const bone = bonesRef.current[boneName]; const orig = origRotRef.current[boneName];
    if (!bone || !orig) return;
    const nat = NATURAL_POSE[boneName] || { x:0,y:0,z:0 };
    const gest = gestureOffset         || { x:0,y:0,z:0 };
    bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, orig.x + nat.x + gest.x, lerpFactor);
    bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, orig.y + nat.y + gest.y, lerpFactor);
    bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, orig.z + nat.z + gest.z, lerpFactor);
  };

  const setBoneOffset = (boneName, ox, oy, oz, lerpFactor) => {
    const bone = bonesRef.current[boneName]; const orig = origRotRef.current[boneName];
    if (!bone || !orig) return;
    bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, orig.x + ox, lerpFactor);
    bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, orig.y + oy, lerpFactor);
    bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, orig.z + oz, lerpFactor);
  };

  useFrame((state, delta) => {
    const lipSync = lipSyncRef?.current;
    const now     = state.clock.elapsedTime * 1000;
    idleTimeRef.current += delta;
    const t = idleTimeRef.current;
    const meshes = meshesRef.current;

    // Speech amplitude
    let rawAmp = 0;
    if (lipSync && isSpeaking) {
      const w = lipSync.getVisemeWeights?.(0.4) || {};
      rawAmp = Object.values(w).reduce((s, v) => s + v, 0) / Math.max(Object.keys(w).length, 1);
    }
    speechAmpRef.current = THREE.MathUtils.lerp(speechAmpRef.current, rawAmp, 0.12);
    const amp = speechAmpRef.current;

    // Gesture system
    gestureCooldownRef.current -= delta;
    if (gestureRef.current) {
      gestureTimeRef.current += delta;
      if (gestureTimeRef.current >= gestureRef.current.duration) {
        gestureRef.current = null; gestureTimeRef.current = 0;
        gestureCooldownRef.current = isSpeaking ? (2 + Math.random() * 3) : (5 + Math.random() * 5);
      }
    } else if (isSpeaking && gestureCooldownRef.current <= 0) {
      gestureRef.current = GESTURE_LIBRARY[Math.floor(Math.random() * GESTURE_LIBRARY.length)];
      gestureTimeRef.current = 0;
    }
    const gestureOffsets = gestureRef.current ? interpolateGesture(gestureRef.current, gestureTimeRef.current) : {};

    // Morph targets
    if (meshes.length > 0) {
      if (lipSync && isSpeaking) {
        const weights = lipSync.getVisemeWeights(0.4);
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary; const infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const [name, weight] of Object.entries(weights)) {
            const idx = dict[name];
            if (idx !== undefined) infl[idx] = THREE.MathUtils.lerp(infl[idx], weight * 0.85, 0.3);
          }
          const jawIdx = dict["jawOpen"];
          if (jawIdx !== undefined) {
            const jaw = (weights["viseme_aa"]||0)*0.3 + (weights["viseme_O"]||0)*0.25 + (weights["viseme_E"]||0)*0.12 + (weights["viseme_U"]||0)*0.18;
            infl[jawIdx] = THREE.MathUtils.lerp(infl[jawIdx], Math.min(jaw, 0.35), 0.2);
          }
        }
      } else {
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary; const infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const key of Object.keys(dict)) {
            if (key.startsWith("viseme_") || key === "jawOpen" || key === "mouthOpen") {
              const idx = dict[key];
              if (idx !== undefined && infl[idx] > 0.003) { infl[idx] *= 0.93; if (infl[idx] < 0.003) infl[idx] = 0; }
            }
          }
        }
      }

      if (now - lastBlinkRef.current > IDLE_BLINK_INTERVAL + Math.random() * 1500) {
        lastBlinkRef.current = now; blinkTimerRef.current = IDLE_BLINK_DURATION;
      }
      if (blinkTimerRef.current > 0) {
        blinkTimerRef.current -= delta * 1000;
        const p = Math.max(0, 1.0 - blinkTimerRef.current / IDLE_BLINK_DURATION);
        const w = Math.sin(p * Math.PI);
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary; const infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const n of ["eyeBlinkLeft","eyeBlinkRight"]) { const idx = dict[n]; if (idx !== undefined) infl[idx] = w * 0.9; }
        }
      }
      if (isSpeaking) {
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary; const infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const n of ["mouthSmileLeft","mouthSmileRight"]) { const idx = dict[n]; if (idx !== undefined) infl[idx] = THREE.MathUtils.lerp(infl[idx], 0.08, 0.025); }
        }
      }
      if (isThinking) {
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary; const infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          const idx = dict["browInnerUp"];
          if (idx !== undefined) infl[idx] = THREE.MathUtils.lerp(infl[idx], 0.15 + Math.sin(t * 1.5) * 0.05, 0.04);
        }
      }
    }

    // Head
    if (isSpeaking) {
      setBoneOffset("Head", Math.sin(t*1.8)*0.016+Math.sin(t*0.7)*0.006, Math.sin(t*1.1)*0.022+Math.cos(t*0.5)*0.007, Math.sin(t*0.9)*0.007, 0.06);
    } else {
      const ii = isThinking ? 0.011 : 0.005;
      setBoneOffset("Head", Math.sin(t*0.40)*ii, Math.sin(t*0.25)*ii*0.6, Math.cos(t*0.35)*ii*0.3, 0.03);
    }
    isSpeaking
      ? setBoneOffset("Neck", Math.sin(t*1.4)*0.006, Math.sin(t*0.9)*0.008, 0, 0.04)
      : setBoneOffset("Neck", Math.sin(t*0.3)*0.003, 0, 0, 0.02);
    setBoneOffset("Spine1", Math.sin(t*0.7)*0.003, 0, 0, 0.03);
    setBoneOffset("Spine2", Math.sin(t*0.7+0.3)*0.002, 0, 0, 0.03);

    // Arms — slow lerp for large z transition (0.06 avoids snapping)
    setBoneWithPose("RightShoulder", gestureOffsets["RightShoulder"], 0.10);
    setBoneWithPose("LeftShoulder",  gestureOffsets["LeftShoulder"],  0.10);
    setBoneWithPose("RightArm",      gestureOffsets["RightArm"],      0.06);
    setBoneWithPose("LeftArm",       gestureOffsets["LeftArm"],       0.06);
    setBoneWithPose("RightForeArm",  gestureOffsets["RightForeArm"],  0.09);
    setBoneWithPose("LeftForeArm",   gestureOffsets["LeftForeArm"],   0.09);

    // Wrists + speech pulse
    const speechPulseR = isSpeaking ? Math.sin(t*4.5+0.0)*amp*0.06 : 0;
    const speechPulseL = isSpeaking ? Math.sin(t*4.5+Math.PI)*amp*0.06 : 0;
    const rwGest = gestureOffsets["RightHand"] || {x:0,y:0,z:0};
    const lwGest = gestureOffsets["LeftHand"]  || {x:0,y:0,z:0};
    const nat_rh = NATURAL_POSE["RightHand"]; const nat_lh = NATURAL_POSE["LeftHand"];
    const orig_rh = origRotRef.current["RightHand"]; const orig_lh = origRotRef.current["LeftHand"];
    const rh = bonesRef.current["RightHand"]; const lh = bonesRef.current["LeftHand"];
    if (rh && orig_rh) {
      rh.rotation.x = THREE.MathUtils.lerp(rh.rotation.x, orig_rh.x+nat_rh.x+rwGest.x+speechPulseR, 0.09);
      rh.rotation.y = THREE.MathUtils.lerp(rh.rotation.y, orig_rh.y+nat_rh.y+rwGest.y, 0.09);
      rh.rotation.z = THREE.MathUtils.lerp(rh.rotation.z, orig_rh.z+nat_rh.z+rwGest.z, 0.09);
    }
    if (lh && orig_lh) {
      lh.rotation.x = THREE.MathUtils.lerp(lh.rotation.x, orig_lh.x+nat_lh.x+lwGest.x+speechPulseL, 0.09);
      lh.rotation.y = THREE.MathUtils.lerp(lh.rotation.y, orig_lh.y+nat_lh.y+lwGest.y, 0.09);
      lh.rotation.z = THREE.MathUtils.lerp(lh.rotation.z, orig_lh.z+nat_lh.z+lwGest.z, 0.09);
    }

    // Fingers
    for (let i = 0; i < FINGER_BONES.length; i++) {
      const name = FINGER_BONES[i]; const bone = bonesRef.current[name]; const orig = origRotRef.current[name];
      if (!bone || !orig) continue;
      const rest = FINGER_REST_CURL[name] || {x:0,y:0,z:0};
      const phase = i * 1.37;
      let fidgetX, fidgetY, fidgetZ;
      if (isSpeaking) {
        const speechCurl = amp * 0.08 * Math.sin(t * 3.5 + phase);
        fidgetX = Math.sin(t*0.9+phase)*0.03 + speechCurl;
        fidgetY = Math.sin(t*0.6+phase*0.7)*0.008;
        fidgetZ = Math.cos(t*0.7+phase*1.1)*0.006;
      } else {
        fidgetX = Math.sin(t*0.35+phase)*0.018;
        fidgetY = Math.sin(t*0.25+phase*0.6)*0.005;
        fidgetZ = Math.cos(t*0.30+phase*1.0)*0.004;
      }
      bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, orig.x+rest.x+fidgetX, 0.04);
      bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, orig.y+rest.y+fidgetY, 0.04);
      bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, orig.z+rest.z+fidgetZ, 0.04);
    }
  });

  return <primitive object={scene} />;
}

function AvatarLoading() {
  const meshRef = useRef();
  useFrame((s) => { if (meshRef.current) meshRef.current.rotation.y = s.clock.elapsedTime * 0.5; });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color="#6366f1" wireframe transparent opacity={0.6} />
    </mesh>
  );
}

export default function AvatarCanvas({ lipSyncRef, isSpeaking, isThinking }) {
  const controlsRef = useRef();
  return (
    <div style={{ width:"100%", height:"100%", background:"radial-gradient(ellipse at 50% 40%, #1e1e3a 0%, #0f0f1a 70%)", position:"relative" }}>
      <Canvas
        gl={{ antialias:true, alpha:true, powerPreference:"high-performance" }}
        dpr={[1,2]} style={{ background:"transparent" }}
        camera={{ fov:28, near:0.01, far:60, position:[0,1.1,4] }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.5;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <ambientLight intensity={1.0} />
        <directionalLight position={[2,3,4]} intensity={1.5} />
        <directionalLight position={[-2,1,2]} intensity={0.6} color="#c4c4ff" />
        <pointLight position={[0,2,3]} intensity={0.5} />
        <Suspense fallback={<AvatarLoading />}>
          <AvatarModel lipSyncRef={lipSyncRef} isSpeaking={isSpeaking} isThinking={isThinking} controlsRef={controlsRef} />
          <ContactShadows position={[0,-0.88,0]} opacity={0.4} scale={5} blur={2.5} />
          <Environment preset="studio" />
        </Suspense>
        <OrbitControls ref={controlsRef} enablePan={false} enableZoom={true} enableRotate={true}
          minDistance={1.8} maxDistance={6} minPolarAngle={Math.PI/5} maxPolarAngle={Math.PI/1.7} />
      </Canvas>
      <div style={{
        position:"absolute", bottom:16, left:"50%", transform:"translateX(-50%)",
        padding:"6px 16px", borderRadius:20, background:"rgba(0,0,0,0.5)",
        backdropFilter:"blur(8px)", fontSize:12, pointerEvents:"none",
        color: isSpeaking?"#818cf8":isThinking?"#a78bfa":"#64748b",
        letterSpacing:0.5, transition:"color 0.3s",
      }}>
        {isSpeaking ? "● Speaking" : isThinking ? "● Thinking…" : "● Listening"}
      </div>
    </div>
  );
}

useGLTF.preload(AVATAR_URL);
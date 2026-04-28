/**
 * AvatarCanvas — face-only render with HeadTTS-driven lip sync.
 *
 * Stripped down from the full-body version: no hand gestures, no nodding,
 * no finger animation. Only face animation runs per frame:
 *   - 15 Oculus visemes (HeadTTS phoneme-accurate timing)
 *   - jaw open derived from open-mouth visemes
 *   - random eye blinks
 *   - small idle head/neck sway
 *   - thinking brow raise
 *   - subtle smile while speaking
 *
 * Camera is locked to a face-tight frame around the GLB's `Head` bone
 * world position, with OrbitControls clamped so the user can pan/tilt
 * but not zoom out far enough to see the body.
 */

import { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";

const AVATAR_URL = "/avatar/model.glb";
const IDLE_BLINK_INTERVAL = 3200;
const IDLE_BLINK_DURATION = 160;

// Tight face frame.  All distances are in metres.
const FACE_FRAME = {
  fov:           24,
  near:          0.01,
  far:           20,
  cameraOffset:  new THREE.Vector3(0, 0.02, 0.55), // forward of head, just below brow line
  targetOffset:  new THREE.Vector3(0, 0.04, 0),    // look at upper-cheek
  minDistance:   0.42,
  maxDistance:   1.10,
  minPolarAngle: Math.PI * 0.32,
  maxPolarAngle: Math.PI * 0.62,
};

function AvatarModel({ lipSyncRef, isSpeaking, isThinking, controlsRef }) {
  const { scene }  = useGLTF(AVATAR_URL);
  const { camera } = useThree();

  const meshesRef     = useRef([]);
  const headBoneRef   = useRef(null);
  const neckBoneRef   = useRef(null);
  const headOrigRef   = useRef(null);
  const neckOrigRef   = useRef(null);
  const blinkTimerRef = useRef(0);
  const lastBlinkRef  = useRef(0);
  const idleTimeRef   = useRef(0);
  const initDone      = useRef(false);

  useEffect(() => {
    const morphMeshes = [];
    let head = null, neck = null;
    scene.traverse((child) => {
      if (child.isMesh && child.morphTargetDictionary) morphMeshes.push(child);
      if (child.isBone) {
        if (child.name === "Head") head = child;
        if (child.name === "Neck") neck = child;
      }
    });
    meshesRef.current = morphMeshes;
    headBoneRef.current = head;
    neckBoneRef.current = neck;
    if (head) headOrigRef.current = { x: head.rotation.x, y: head.rotation.y, z: head.rotation.z };
    if (neck) neckOrigRef.current = { x: neck.rotation.x, y: neck.rotation.y, z: neck.rotation.z };

    if (!initDone.current && head) {
      initDone.current = true;

      // Resolve head world position once — frame the camera there.
      head.updateWorldMatrix(true, false);
      const headPos = new THREE.Vector3();
      head.getWorldPosition(headPos);

      const target = headPos.clone().add(FACE_FRAME.targetOffset);
      const camPos = headPos.clone().add(FACE_FRAME.cameraOffset);

      camera.position.copy(camPos);
      camera.fov  = FACE_FRAME.fov;
      camera.near = FACE_FRAME.near;
      camera.far  = FACE_FRAME.far;
      camera.lookAt(target);
      camera.updateProjectionMatrix();

      if (controlsRef?.current) {
        const c = controlsRef.current;
        c.target.copy(target);
        c.minDistance   = FACE_FRAME.minDistance;
        c.maxDistance   = FACE_FRAME.maxDistance;
        c.minPolarAngle = FACE_FRAME.minPolarAngle;
        c.maxPolarAngle = FACE_FRAME.maxPolarAngle;
        c.enablePan     = false;
        c.update();
      }
      console.log("[Avatar] face-frame init  head=", headPos.toArray().map(n => n.toFixed(2)));
    }
  }, [scene, camera, controlsRef]);

  useFrame((state, delta) => {
    const lipSync = lipSyncRef?.current;
    const now     = state.clock.elapsedTime * 1000;
    idleTimeRef.current += delta;
    const t = idleTimeRef.current;
    const meshes = meshesRef.current;

    // ── Viseme weights once per frame (slower chase for natural speech rhythm) ──
    const visemeWeights = (lipSync && isSpeaking)
      ? lipSync.getVisemeWeights(0.20)
      : null;

    // ── Morph targets ──
    // The lip-sync module now returns a full HeadTTS-style articulator map:
    //   viseme_* (only the top-2 dominant phonemes carry weight)
    //   jawOpen, mouthClose, mouthFunnel, mouthPucker, mouthSmileLeft/Right
    // We just apply each target name directly.  Closures (PP/FF/M) drive
    // jawOpen→0 + mouthClose high; rounded vowels (O/U) engage funnel+pucker;
    // front vowels (E/I) engage smile.
    const MOUTH_TARGETS = [
      "viseme_sil","viseme_PP","viseme_FF","viseme_TH","viseme_DD","viseme_kk",
      "viseme_CH","viseme_SS","viseme_nn","viseme_RR",
      "viseme_aa","viseme_E","viseme_I","viseme_O","viseme_U",
      "jawOpen","mouthClose","mouthFunnel","mouthPucker",
      "mouthSmileLeft","mouthSmileRight",
    ];
    if (meshes.length > 0) {
      if (visemeWeights) {
        const w = visemeWeights;
        // Per-target lerps tuned for natural speech motion.  Jaw is the
        // heaviest articulator (slowest), then mouthClose (closures),
        // then lip shapes (funnel/pucker/smile), then phoneme morphs.
        // Compounds with lipSync's internal lerp ≈ 0.20.
        const SLOW_TARGETS = {
          jawOpen:        0.10,   // ~125ms to 90% — natural jaw rate
          mouthClose:     0.14,   // closures resolve gracefully
          mouthFunnel:    0.18,
          mouthPucker:    0.18,
          mouthSmileLeft: 0.20,
          mouthSmileRight:0.20,
        };
        const FAST_LERP = 0.24;   // viseme_* phoneme morphs
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary, infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const name of MOUTH_TARGETS) {
            const idx = dict[name];
            if (idx === undefined) continue;
            const weight = w[name] || 0;
            const lf = SLOW_TARGETS[name] ?? FAST_LERP;
            infl[idx] = THREE.MathUtils.lerp(infl[idx], weight, lf);
          }
        }
      } else {
        // Decay any leftover mouth weights to zero
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary, infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const name of MOUTH_TARGETS) {
            const idx = dict[name];
            if (idx !== undefined && infl[idx] > 0.003) {
              infl[idx] *= 0.92;
              if (infl[idx] < 0.003) infl[idx] = 0;
            }
          }
        }
      }

      // Blinks
      if (now - lastBlinkRef.current > IDLE_BLINK_INTERVAL + Math.random() * 1500) {
        lastBlinkRef.current = now;
        blinkTimerRef.current = IDLE_BLINK_DURATION;
      }
      if (blinkTimerRef.current > 0) {
        blinkTimerRef.current -= delta * 1000;
        const p = Math.max(0, 1.0 - blinkTimerRef.current / IDLE_BLINK_DURATION);
        const blinkW = Math.sin(p * Math.PI);
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary, infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          for (const n of ["eyeBlinkLeft","eyeBlinkRight"]) {
            const idx = dict[n];
            if (idx !== undefined) infl[idx] = blinkW * 0.9;
          }
        }
      }

      // (Smile while speaking is now driven by the lip-sync articulation
      // table — viseme_E and viseme_I add a touch of mouthSmile.)

      // Brow raise on thinking
      if (isThinking) {
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary, infl = mesh.morphTargetInfluences;
          if (!dict || !infl) continue;
          const idx = dict["browInnerUp"];
          if (idx !== undefined) {
            infl[idx] = THREE.MathUtils.lerp(infl[idx], 0.13 + Math.sin(t * 1.5) * 0.04, 0.04);
          }
        }
      }
    }

    // ── Very subtle head + neck sway only (no nodding, no body bones) ──
    const head = headBoneRef.current, headOrig = headOrigRef.current;
    if (head && headOrig) {
      const ampX = isSpeaking ? 0.008 : 0.004;
      const ampY = isSpeaking ? 0.012 : 0.005;
      const ampZ = isSpeaking ? 0.004 : 0.002;
      const tx = Math.sin(t * 0.65) * ampX + Math.sin(t * 0.21) * ampX * 0.4;
      const ty = Math.sin(t * 0.41) * ampY + Math.cos(t * 0.27) * ampY * 0.3;
      const tz = Math.cos(t * 0.55) * ampZ;
      head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, headOrig.x + tx, 0.04);
      head.rotation.y = THREE.MathUtils.lerp(head.rotation.y, headOrig.y + ty, 0.04);
      head.rotation.z = THREE.MathUtils.lerp(head.rotation.z, headOrig.z + tz, 0.04);
    }
    const neck = neckBoneRef.current, neckOrig = neckOrigRef.current;
    if (neck && neckOrig) {
      const a = isSpeaking ? 0.003 : 0.0015;
      const tx = Math.sin(t * 0.55) * a;
      const ty = Math.sin(t * 0.33) * a * 0.5;
      neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, neckOrig.x + tx, 0.03);
      neck.rotation.y = THREE.MathUtils.lerp(neck.rotation.y, neckOrig.y + ty, 0.03);
    }
  });

  return <primitive object={scene} />;
}

function AvatarLoading() {
  const meshRef = useRef();
  useFrame((s) => { if (meshRef.current) meshRef.current.rotation.y = s.clock.elapsedTime * 0.5; });
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.15, 24, 24]} />
      <meshStandardMaterial color="#4f46e5" wireframe transparent opacity={0.35} />
    </mesh>
  );
}

export default function AvatarCanvas({ lipSyncRef, isSpeaking, isThinking }) {
  const controlsRef = useRef();
  return (
    <div style={{
      width:"100%", height:"100%", position:"relative",
      background: "radial-gradient(ellipse at 50% 35%, #ffffff 0%, #eef2f9 55%, #dde4f0 100%)",
    }}>
      <Canvas
        gl={{ antialias:true, alpha:true, powerPreference:"high-performance" }}
        dpr={[1,2]} style={{ background:"transparent" }}
        camera={{ fov: FACE_FRAME.fov, near: FACE_FRAME.near, far: FACE_FRAME.far, position:[0, 1.6, 0.6] }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <ambientLight intensity={0.65} />
        <hemisphereLight args={["#ffffff", "#dde4f0", 0.55]} />
        <directionalLight position={[1.5, 2.0, 1.5]} intensity={1.5} color="#fff8ec" />
        <directionalLight position={[-1.5, 1.5, 1.0]} intensity={0.55} color="#d4dcf2" />
        <pointLight position={[0, 1.8, 1.5]} intensity={0.30} color="#ffffff" />
        <Suspense fallback={<AvatarLoading />}>
          <AvatarModel lipSyncRef={lipSyncRef} isSpeaking={isSpeaking} isThinking={isThinking} controlsRef={controlsRef} />
          <ContactShadows position={[0, -0.88, 0]} opacity={0.18} scale={3} blur={3.5} far={2} />
          <Environment preset="apartment" />
        </Suspense>
        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={FACE_FRAME.minDistance}
          maxDistance={FACE_FRAME.maxDistance}
          minPolarAngle={FACE_FRAME.minPolarAngle}
          maxPolarAngle={FACE_FRAME.maxPolarAngle}
        />
      </Canvas>
    </div>
  );
}

useGLTF.preload(AVATAR_URL);

/**
 * AvatarCanvas — Three.js 3D avatar with real-time lip sync.
 *
 * Renders a GLB avatar using @react-three/fiber and drives lip
 * sync via morph targets (if available) or jaw bone rotation.
 * Auto-computes bounding box to center the camera on the face.
 *
 * Props:
 *   lipSyncRef  — ref to LipSyncAudioQueue instance
 *   isSpeaking  — whether audio is currently playing
 *   isThinking  — whether the agent is generating
 */

import { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AVATAR_URL = "/avatar/model.glb";
const IDLE_BLINK_INTERVAL = 3500;
const IDLE_BLINK_DURATION = 150;

// Bone names to search for (various avatar conventions)
const JAW_BONE_NAMES = ["Jaw", "jaw", "JawBone", "jawBone", "jaw_bone"];
const HEAD_BONE_NAMES = [
  "Head", "head", "Head_01", "Head_00",
  "mixamorigHead", "HeadTop_End",
];

// ---------------------------------------------------------------------------
// Avatar Model — inner Three.js component
// ---------------------------------------------------------------------------
function AvatarModel({ lipSyncRef, isSpeaking, isThinking }) {
  const { scene } = useGLTF(AVATAR_URL);
  const { camera } = useThree();
  const meshesRef = useRef([]);
  const blinkTimerRef = useRef(0);
  const lastBlinkRef = useRef(0);
  const headBoneRef = useRef(null);
  const jawBoneRef = useRef(null);
  const idleTimeRef = useRef(0);
  const hasMorphTargets = useRef(false);
  const cameraReady = useRef(false);

  // On mount: find meshes/bones and position camera
  useEffect(() => {
    const morphMeshes = [];
    let foundHead = false;
    let foundJaw = false;

    scene.traverse((child) => {
      if (child.isMesh && child.morphTargetDictionary) {
        morphMeshes.push(child);
        console.log(
          `[Avatar] Mesh "${child.name}" morph targets:`,
          Object.keys(child.morphTargetDictionary).sort()
        );
      }

      if (child.isBone && !foundHead) {
        for (const name of HEAD_BONE_NAMES) {
          if (child.name === name || child.name.includes(name)) {
            headBoneRef.current = child;
            foundHead = true;
            console.log(`[Avatar] Head bone found: "${child.name}"`);
            break;
          }
        }
      }

      if (child.isBone && !foundJaw) {
        for (const name of JAW_BONE_NAMES) {
          if (child.name === name || child.name.includes(name)) {
            jawBoneRef.current = child;
            foundJaw = true;
            console.log(`[Avatar] Jaw bone found: "${child.name}"`);
            break;
          }
        }
      }
    });

    meshesRef.current = morphMeshes;
    hasMorphTargets.current = morphMeshes.length > 0;

    if (morphMeshes.length === 0) {
      console.warn("[Avatar] No meshes with morph targets found in GLB!");
      if (!foundJaw) {
        console.warn("[Avatar] No jaw bone found either — lip sync will use head micro-nod");
      }
    } else {
      console.log(`[Avatar] Found ${morphMeshes.length} mesh(es) with morph targets`);
    }

    // --- Auto-fit camera to the model's head ---
    if (!cameraReady.current) {
      cameraReady.current = true;

      const box = new THREE.Box3().setFromObject(scene);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);

      console.log(
        `[Avatar] Bounds: min=(${box.min.x.toFixed(2)}, ${box.min.y.toFixed(2)}, ${box.min.z.toFixed(2)}) ` +
        `max=(${box.max.x.toFixed(2)}, ${box.max.y.toFixed(2)}, ${box.max.z.toFixed(2)}) ` +
        `size=(${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)})`
      );

      // Focus on full body: frame the entire model
      const lookAtY = center.y;

      // Calculate distance to frame the full body
      const fov = 30;
      const halfFovRad = THREE.MathUtils.degToRad(fov / 2);
      const frameHeight = size.y * 1.15;
      const dist = (frameHeight / 2) / Math.tan(halfFovRad);

      camera.position.set(center.x, lookAtY, center.z + Math.max(dist, 2.5));
      camera.lookAt(center.x, lookAtY, center.z);
      camera.fov = fov;
      camera.near = 0.01;
      camera.far = 50;
      camera.updateProjectionMatrix();

      console.log(
        `[Avatar] Camera set: pos=(${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)}) ` +
        `target=(${center.x.toFixed(2)}, ${headY.toFixed(2)}, ${center.z.toFixed(2)}) fov=${fov}`
      );
    }
  }, [scene, camera]);

  // Per-frame update
  useFrame((state, delta) => {
    const lipSync = lipSyncRef?.current;
    const now = state.clock.elapsedTime * 1000;
    idleTimeRef.current += delta;

    // --- Lip sync via morph targets ---
    if (hasMorphTargets.current) {
      const meshes = meshesRef.current;

      if (lipSync && isSpeaking) {
        const weights = lipSync.getVisemeWeights(0.35);
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary;
          const influences = mesh.morphTargetInfluences;
          if (!dict || !influences) continue;
          for (const [name, weight] of Object.entries(weights)) {
            const idx = dict[name];
            if (idx !== undefined) {
              influences[idx] = THREE.MathUtils.lerp(influences[idx], weight, 0.3);
            }
          }
        }
      } else {
        for (const mesh of meshes) {
          const dict = mesh.morphTargetDictionary;
          const influences = mesh.morphTargetInfluences;
          if (!dict || !influences) continue;
          for (const key of Object.keys(dict)) {
            if (key.startsWith("viseme")) {
              const idx = dict[key];
              if (idx !== undefined && influences[idx] > 0.01) {
                influences[idx] *= 0.88;
                if (influences[idx] < 0.01) influences[idx] = 0;
              }
            }
          }
        }
      }
    }

    // --- Lip sync via jaw bone (fallback when no morph targets) ---
    if (!hasMorphTargets.current && jawBoneRef.current) {
      if (lipSync && isSpeaking) {
        const weights = lipSync.getVisemeWeights(0.35);
        const jawOpen =
          (weights.viseme_aa || 0) * 0.8 +
          (weights.viseme_O || 0) * 0.6 +
          (weights.viseme_E || 0) * 0.4 +
          (weights.viseme_U || 0) * 0.5 +
          (weights.viseme_I || 0) * 0.3;
        const clampedJaw = Math.min(1.0, jawOpen) * 0.15;
        jawBoneRef.current.rotation.x = THREE.MathUtils.lerp(
          jawBoneRef.current.rotation.x,
          clampedJaw,
          0.3
        );
      } else {
        jawBoneRef.current.rotation.x = THREE.MathUtils.lerp(
          jawBoneRef.current.rotation.x,
          0,
          0.15
        );
      }
    }

    // --- Lip sync via head micro-nod (last resort fallback) ---
    if (!hasMorphTargets.current && !jawBoneRef.current && headBoneRef.current) {
      if (lipSync && isSpeaking) {
        const weights = lipSync.getVisemeWeights(0.35);
        const energy =
          (weights.viseme_aa || 0) + (weights.viseme_O || 0) +
          (weights.viseme_E || 0) + (weights.viseme_U || 0);
        const nod = Math.min(1.0, energy) * 0.03;
        headBoneRef.current.rotation.x = THREE.MathUtils.lerp(
          headBoneRef.current.rotation.x,
          nod,
          0.25
        );
      }
    }

    // --- Idle blink (only for models with morph targets) ---
    if (hasMorphTargets.current) {
      if (now - lastBlinkRef.current > IDLE_BLINK_INTERVAL + Math.random() * 1000) {
        lastBlinkRef.current = now;
        blinkTimerRef.current = IDLE_BLINK_DURATION;
      }

      if (blinkTimerRef.current > 0) {
        blinkTimerRef.current -= delta * 1000;
        const blinkProgress = 1.0 - blinkTimerRef.current / IDLE_BLINK_DURATION;
        const blinkWeight = Math.sin(blinkProgress * Math.PI);

        for (const mesh of meshesRef.current) {
          const dict = mesh.morphTargetDictionary;
          const influences = mesh.morphTargetInfluences;
          if (!dict || !influences) continue;
          for (const blinkName of [
            "eyeBlinkLeft", "eyeBlinkRight", "eyesClosed",
            "EyeBlink_L", "EyeBlink_R",
          ]) {
            const idx = dict[blinkName];
            if (idx !== undefined) {
              influences[idx] = blinkWeight * 0.9;
            }
          }
        }
      }
    }

    // --- Subtle idle head movement ---
    if (headBoneRef.current && !isSpeaking) {
      const t = idleTimeRef.current;
      const intensity = isThinking ? 0.02 : 0.008;
      headBoneRef.current.rotation.x += Math.sin(t * 0.5) * intensity * delta;
      headBoneRef.current.rotation.y = Math.sin(t * 0.3) * intensity * 0.5;
      headBoneRef.current.rotation.z = Math.cos(t * 0.4) * intensity * 0.3;
    }
  });

  return <primitive object={scene} />;
}

// ---------------------------------------------------------------------------
// Loading fallback
// ---------------------------------------------------------------------------
function AvatarLoading() {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial
        color="#6366f1"
        wireframe
        transparent
        opacity={0.6}
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Exported Canvas wrapper
// ---------------------------------------------------------------------------
export default function AvatarCanvas({ lipSyncRef, isSpeaking, isThinking }) {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "radial-gradient(ellipse at 50% 40%, #1a1a2e 0%, #0f0f1a 70%)",
        border: `2px solid ${
          isSpeaking
            ? "rgba(99,102,241,0.6)"
            : isThinking
            ? "rgba(99,102,241,0.3)"
            : "rgba(255,255,255,0.08)"
        }`,
        boxShadow: isSpeaking
          ? "0 0 30px rgba(99,102,241,0.3)"
          : isThinking
          ? "0 0 15px rgba(99,102,241,0.15)"
          : "none",
        transition: "border-color 0.5s, box-shadow 0.5s",
        position: "relative",
      }}
    >
      <Canvas
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        dpr={[1, 2]}
        style={{ background: "transparent" }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.5;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        {/* Lighting — bright enough to ensure avatar is visible */}
        <ambientLight intensity={1.2} />
        <directionalLight position={[2, 3, 4]} intensity={1.5} />
        <directionalLight position={[-2, 1, 2]} intensity={0.8} color="#c4c4ff" />
        <pointLight position={[0, 2, 3]} intensity={0.6} color="#ffffff" />

        <Suspense fallback={<AvatarLoading />}>
          <AvatarModel
            lipSyncRef={lipSyncRef}
            isSpeaking={isSpeaking}
            isThinking={isThinking}
          />
          <ContactShadows position={[0, -0.88, 0]} opacity={0.4} scale={5} blur={2.5} />
          <Environment preset="studio" />
        </Suspense>

        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={1.5}
          maxDistance={8}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 1.6}
        />
      </Canvas>
    </div>
  );
}

// Preload the GLB model
useGLTF.preload(AVATAR_URL);

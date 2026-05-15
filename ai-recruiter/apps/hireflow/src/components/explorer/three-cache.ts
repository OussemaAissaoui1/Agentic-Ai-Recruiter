// Three.js shared resources for the 3D knowledge graph.
//
// Without these caches, every halo state-change in the force-graph would
// allocate a fresh SphereGeometry + MeshLambertMaterial per node and churn the
// GC during pulse animations. The number of unique (type, halo) pairs is
// bounded at 16 (8 node types × halo on/off), so we precompute once and reuse.

import * as THREE from "three";

import type { JDGraphNode } from "@/lib/api";

// Color per node type — keep in sync with NODE_TYPE_BY_LABEL in
// agents/jd_generation/graph/dump.py on the backend.
export const NODE_COLOR: Record<JDGraphNode["type"], string> = {
  employee:        "#a78bfa", // violet-400
  role:            "#5eead4", // teal-300
  skill:           "#fbbf24", // amber-400
  team:            "#94a3b8", // slate-400
  education:       "#fdba74", // orange-300
  prior_company:   "#fb7185", // rose-400
  job_description: "#c084fc", // violet-400 brighter
  rejection:       "#ef4444", // red-500
};

export const HALO_COLOR = "#facc15"; // yellow-400
export const HALO_DURATION_MS = 3500;

const NODE_GEO_DEFAULT = new THREE.SphereGeometry(5, 16, 12);
const NODE_GEO_HALO = new THREE.SphereGeometry(7, 16, 12);

const NODE_MATERIALS = new Map<string, THREE.MeshLambertMaterial>();

function makeMaterial(color: string, halo: boolean): THREE.MeshLambertMaterial {
  return new THREE.MeshLambertMaterial({
    color,
    emissive: halo ? HALO_COLOR : "#000000",
    emissiveIntensity: halo ? 0.45 : 0,
    transparent: true,
    opacity: halo ? 1.0 : 0.95,
  });
}

export function nodeMeshFor(type: string, halo: boolean): THREE.Mesh {
  const color = halo ? HALO_COLOR : NODE_COLOR[type as keyof typeof NODE_COLOR] ?? "#888888";
  const key = `${halo ? "h" : "n"}:${color}`;
  let mat = NODE_MATERIALS.get(key);
  if (!mat) {
    mat = makeMaterial(color, halo);
    NODE_MATERIALS.set(key, mat);
  }
  return new THREE.Mesh(halo ? NODE_GEO_HALO : NODE_GEO_DEFAULT, mat);
}

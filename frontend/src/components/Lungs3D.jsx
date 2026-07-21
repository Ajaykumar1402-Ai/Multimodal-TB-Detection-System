import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

function LungLayer({ count, color, size, position, scale, speed }) {
  const points = useRef();
  
  // Create an anatomical lung-like shape using two ellipsoids
  const [leftLung, rightLung] = useMemo(() => {
    const left = new Float32Array(count * 3);
    const right = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      // Left Lung (Ellipsoid)
      const u = Math.random() * Math.PI * 2;
      const v = Math.random() * Math.PI;
      left[i * 3] = 0.8 * Math.cos(u) * Math.sin(v) - 0.7;
      left[i * 3 + 1] = 1.5 * Math.sin(u) * Math.sin(v);
      left[i * 3 + 2] = 0.6 * Math.cos(v);

      // Right Lung (Ellipsoid)
      right[i * 3] = 0.8 * Math.cos(u) * Math.sin(v) + 0.7;
      right[i * 3 + 1] = 1.5 * Math.sin(u) * Math.sin(v);
      right[i * 3 + 2] = 0.6 * Math.cos(v);
    }
    return [left, right];
  }, [count]);

  useFrame((state) => {
    const t = state.clock.getElapsedTime() * speed;
    // Breathing pulse
    const s = 1 + Math.sin(t) * 0.05;
    points.current.scale.set(s, s, s);
    points.current.rotation.y += 0.002;
  });

  return (
    <group ref={points} position={position} scale={scale}>
      <Points positions={leftLung}>
        <PointMaterial transparent color={color} size={size} sizeAttenuation={true} depthWrite={false} blending={THREE.AdditiveBlending} />
      </Points>
      <Points positions={rightLung}>
        <PointMaterial transparent color={color} size={size} sizeAttenuation={true} depthWrite={false} blending={THREE.AdditiveBlending} />
      </Points>
    </group>
  );
}

export default function Lungs3D() {
  return (
    <div className="w-full h-full cursor-pointer">
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <color attach="background" args={['transparent']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
          {/* Inner Core */}
          <LungLayer count={1500} color="#3b82f6" size={0.06} position={[0, 0, 0]} scale={[1, 1, 1]} speed={1.5} />
          {/* Outer Atmosphere */}
          <LungLayer count={800} color="#6366f1" size={0.12} position={[0, 0, 0]} scale={[1.2, 1.2, 1.2]} speed={0.8} />
        </Float>
      </Canvas>
    </div>
  );
}

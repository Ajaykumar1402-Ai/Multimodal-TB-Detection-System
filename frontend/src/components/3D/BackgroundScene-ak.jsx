import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';

const AnimatedSphere = ({ mouse }) => {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      // Smooth rotation
      meshRef.current.rotation.x = state.clock.getElapsedTime() * 0.1;
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.15;
      
      // Gentle mouse follow (parallax)
      meshRef.current.position.x = THREE.MathUtils.lerp(meshRef.current.position.x, mouse.current[0] * 0.5, 0.05);
      meshRef.current.position.y = THREE.MathUtils.lerp(meshRef.current.position.y, mouse.current[1] * 0.5, 0.05);
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <Sphere args={[1, 32, 64]} scale={2.2} ref={meshRef}>
        <MeshDistortMaterial
          color="#0ea5e9"
          attach="material"
          distort={0.45}
          speed={2}
          roughness={0.1}
          metalness={0.9}
          transparent
          opacity={0.8}
        />
      </Sphere>
    </Float>
  );
};

const Particles = ({ count, mouse }) => {
  const mesh = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      const t = Math.random() * 100;
      const factor = 20 + Math.random() * 100;
      const speed = 0.005 + Math.random() / 500;
      const xFactor = -50 + Math.random() * 100;
      const yFactor = -50 + Math.random() * 100;
      const zFactor = -50 + Math.random() * 100;
      temp.push({ t, factor, speed, xFactor, yFactor, zFactor });
    }
    return temp;
  }, [count]);

  useFrame((state) => {
    if (!mesh.current) return;
    particles.forEach((particle, i) => {
      let { t, factor, speed, xFactor, yFactor, zFactor } = particle;
      t = particle.t += speed;
      const s = Math.cos(t);
      
      const x = xFactor + Math.cos((t / 10) * factor) + (Math.sin(t * 1) * factor) / 10 + (mouse.current[0] * 1.5);
      const y = yFactor + Math.sin((t / 10) * factor) + (Math.cos(t * 2) * factor) / 10 + (mouse.current[1] * 1.5);
      const z = zFactor + Math.cos((t / 10) * factor) + (Math.sin(t * 3) * factor) / 10;

      dummy.position.set(x, y, z);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      mesh.current.setMatrixAt(i, dummy.matrix);
    });
    mesh.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[null, null, count]}>
      <sphereGeometry args={[0.06, 6, 6]} />
      <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} />
    </instancedMesh>
  );
};

export default function BackgroundScene() {
  const mouse = useRef([0, 0]);

  useEffect(() => {
    const handleMouseMove = (event) => {
      mouse.current = [
        (event.clientX / window.innerWidth) * 2 - 1,
        -(event.clientY / window.innerHeight) * 2 + 1,
      ];
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="w-full h-full opacity-40">
      <Canvas camera={{ position: [0, 0, 15], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={0.4} />
        <AnimatedSphere mouse={mouse} />
        <Particles count={50} mouse={mouse} />
      </Canvas>
    </div>
  );
}

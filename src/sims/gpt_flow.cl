#define IDX(x, y, width) ((y) * (width) + (x))

// Sample divergence of velocity field
__kernel void computeDivergence(
   __global const float2* velocity,
   __global float* divergence,
   __global const float* solidMask,
   int width,
   int height,
   float cellSize)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int idx = IDX(x, y, width);

   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
   if (solidMask[idx] > 0.5f) {
      divergence[idx] = 0.0f;
      return;
   }

   float2 velL = velocity[IDX(x - 1, y, width)];
   float2 velR = velocity[IDX(x + 1, y, width)];
   float2 velB = velocity[IDX(x, y - 1, width)];
   float2 velT = velocity[IDX(x, y + 1, width)];

   float div = (velR.x - velL.x + velT.y - velB.y) / (2.0f * cellSize);
   divergence[idx] = div;
}


// Jacobi iteration for pressure solving
__kernel void pressureJacobi(
   __global const float* divergence,
   __global const float* pressureIn,
   __global float* pressureOut,
   __global const float* solidMask,
   int width,
   int height,
   float alpha,
   float rBeta)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
   if (solidMask[idx] > 0.5f) {
      pressureOut[idx] = 0.0f;
      return;
   }

   float pL = pressureIn[IDX(x - 1, y, width)];
   float pR = pressureIn[IDX(x + 1, y, width)];
   float pB = pressureIn[IDX(x, y - 1, width)];
   float pT = pressureIn[IDX(x, y + 1, width)];

   float b = divergence[idx];

   pressureOut[idx] = (pL + pR + pB + pT - alpha * b) * rBeta;
}

// Subtract pressure gradient from velocity
__kernel void subtractPressureGradient(
   __global const float* pressure,
   __global float2* velocity,
   __global const float* solidMask,
   int width,
   int height,
   float cellSize)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
   if (solidMask[idx] > 0.5f) {
      velocity[idx] = 0.0f;
      return;
   }

   float pL = pressure[IDX(x - 1, y, width)];
   float pR = pressure[IDX(x + 1, y, width)];
   float pB = pressure[IDX(x, y - 1, width)];
   float pT = pressure[IDX(x, y + 1, width)];

   float2 vel = velocity[idx];
   vel.x -= (pR - pL) / (2.0f * cellSize);
   vel.y -= (pT - pB) / (2.0f * cellSize);
   velocity[idx] = vel;
}

// Advect velocity using semi-Lagrangian method
__kernel void advectVelocity(
   __global const float2* velocityIn,
   __global float2* velocityOut,
   __global const float* solidMask,
   int width,
   int height,
   float dt,
   float cellSize)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (solidMask[idx] > 0.5f) {
      velocityOut[idx] = (float2)(0.0f, 0.0f); // block motion
      return;
   }
   float2 pos = (float2)(x, y);

   // Trace back
   float2 vel = velocityIn[idx];
   float2 prevPos = pos - (dt / cellSize) * vel;

   // Clamp for bounds
   prevPos.x = clamp(prevPos.x, 0.0f, (float)(width - 1));
   prevPos.y = clamp(prevPos.y, 0.0f, (float)(height - 1));

   // Bilinear sample
   int x0 = clamp((int)floor(prevPos.x), 0, width - 2);
   int y0 = clamp((int)floor(prevPos.y), 0, height - 2);
   float sx = prevPos.x - x0;
   float sy = prevPos.y - y0;

   float2 v00 = velocityIn[IDX(x0, y0, width)];
   float2 v10 = velocityIn[IDX(x0 + 1, y0, width)];
   float2 v01 = velocityIn[IDX(x0, y0 + 1, width)];
   float2 v11 = velocityIn[IDX(x0 + 1, y0 + 1, width)];

   float2 vx0 = mix(v00, v10, sx);
   float2 vx1 = mix(v01, v11, sx);
   float2 sampled = mix(vx0, vx1, sy);

   velocityOut[idx] = sampled;
}

// Apply inflow/outflow boundary conditions
__kernel void applyBoundary(
   __global float2* velocity,
   __global const float* solidMask,
   int width,
   int height,
   float inflowVelocity)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (solidMask[idx] > 0.5f) {
      velocity[idx] = (float2)(0.0f, 0.0f);
      return;
   }
   // Inlet: left middle third
   // if (x == 1 && y > height / 3 && y < 2 * height / 3) {
   if (x == 0) {
      velocity[idx] = (float2)(inflowVelocity, 0.0f);
   }

   // Outlet: right edge
   if (x > width - 1) {
      velocity[idx] = (float2)(0.0f, 0.0f);
   }
   
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
}
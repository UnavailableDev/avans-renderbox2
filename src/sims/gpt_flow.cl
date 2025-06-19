#define IDX(x, y, width) ((y) * (width) + (x))

/* 
   0,0                  width,0
      +-----------------+
      |                 |
      |                 |
   In |       -->       | Out
      |                 |
      |                 |
      +-----------------+
   0,height             width,height


*/


// Sample divergence of velocity field
__kernel void computeDivergence(
   __global const float2* velocity,
   __global float* divergence,
   __global const float* solidMask,
   int width,
   int height,
   float cellSize)
{
/// OKAY
   int x = get_global_id(0);
   int y = get_global_id(1);
   int idx = IDX(x, y, width);

   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
   if (solidMask[idx] > 0.5f) {
      divergence[idx] = 0.0f;
      return;
   }

   // float2 velL = velocity[IDX(x - 1, y, width)];
   // float2 velR = velocity[IDX(x + 1, y, width)];
   // float2 velB = velocity[IDX(x, y - 1, width)];
   // float2 velT = velocity[IDX(x, y + 1, width)];

   float2 velL = velocity[IDX(clamp(x - 1, 0 , width), y, width)];
   float2 velR = velocity[IDX(clamp(x + 1, 0 , width), y, width)];
   float2 velB = velocity[IDX(x, clamp(y + 1, 0, height), width)];
   float2 velT = velocity[IDX(x, clamp(y - 1, 0, height), width)];

   float div = (velR.x - velL.x + velT.y - velB.y) / (2.0f * cellSize);
   // float div = (velR.x - velL.x + velT.y - velB.y);
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

   float pL = pressureIn[IDX(clamp(x - 1, 0, width), y, width)];
   float pR = pressureIn[IDX(clamp(x + 1, 0, width), y, width)];
   float pB = pressureIn[IDX(x, clamp(y + 1, 0, height), width)];
   float pT = pressureIn[IDX(x, clamp(y - 1, 0, height), width)];

   float b = divergence[idx];

   float cellSize = 1.0f;
   // pressureOut[idx] = 0.25f * (pL + pR + pB + pT - exp2(cellSize) * b);
   float pressure = 0.25f * (pL + pR + pB + pT - exp2(cellSize) * b);
   pressureOut[idx] = pressure - 0.1f * pressure ;
   // pressureOut[idx] = (pL + pR + pB + pT - alpha * b) * rBeta;
   // if (y == 50) pressureOut[idx] = 30.0f; // Boundary condition: top row is zero pressure
}

// Make fluid simulation incompressible by subtracting pressure gradient from velocity
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

   float pL = pressure[IDX(clamp(x - 1, 0, width), y, width)];
   float pR = pressure[IDX(clamp(x + 1, 0, width), y, width)];
   float pB = pressure[IDX(x, clamp(y + 1, 0, height), width)];
   float pT = pressure[IDX(x, clamp(y - 1, 0, height), width)];

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
__kernel void applyForce(
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
   
   // if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) velocity[idx] = (float2)(0.0f, 0.0f); // Removes access velocity at end
   if (x == 0 /*|| x == width - 1*/) {
      velocity[idx] = (float2)(inflowVelocity, 0.0f);
   }
}

__kernel void smoothPressure(
   __global const float* pressureIn,
   __global float* pressureOut,
   __global const float* solidMask,
   int width,
   int height,
   float cellSize
) {
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;
   if (solidMask[idx] > 0.5f) {
      pressureOut[idx] = 0.0f;
      return;
   }

   float pTL = pressureIn[IDX(clamp(x - 1, 0, width), clamp(y - 1, 0, height), width)];
   float pTR = pressureIn[IDX(clamp(x + 1, 0, width), clamp(y - 1, 0, height), width)];
   float pBL = pressureIn[IDX(clamp(x - 1, 0, width), clamp(y + 1, 0, height), width)];
   float pBR = pressureIn[IDX(clamp(x + 1, 0, width), clamp(y + 1, 0, height), width)];

   float pL = pressureIn[IDX(clamp(x - 1, 0, width), y, width)];
   float pR = pressureIn[IDX(clamp(x + 1, 0, width), y, width)];
   float pB = pressureIn[IDX(x, clamp(y + 1, 0, height), width)];
   float pT = pressureIn[IDX(x, clamp(y - 1, 0, height), width)];

   // pressureOut[idx] = (pL + pR + pB + pT + pTL + pTR + pBL + pBR) / 8; // Simple averaging
   pressureOut[idx] = pressureIn[idx] + 0.2f * (pL + pR + pB + pT - 4 * pressureIn[idx]); // Laplacian smoothing
}

__kernel void abs_velocity(
   __global const float2* velocityIn,
   __global float* velocityOut,
   int width,
   int height
) {
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;

   // velocityOut[idx] = (velocityIn[idx].y);
   velocityOut[idx] = log10( sqrt(exp2(velocityIn[idx].x) + exp2(velocityIn[idx].y)) )/2; //length of velocity vector
   // velocityOut[idx] = (velocityIn[idx].x) + (velocityIn[idx].y); //float abs value of velocity
}

__kernel void g_velocity(
   __global const float2* velocityIn,
   __global uchar3* rgb_out,
   int width,
   int height
) {
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   if (x <= 0 || y <= 0 || x >= width-1 || y >= height-1) return;

   // velocityOut[idx] = (velocityIn[idx].y);
   float vel = log10( sqrt(exp2(velocityIn[idx].x) + exp2(velocityIn[idx].y)) ); //length of velocity vector
   rgb_out[idx] = (uchar3)(
      (uchar)(clamp(vel * 128.0f, 0.0f, 255.0f)), // Red channel
      (uchar)(clamp(vel * 128.0f, 0.0f, 255.0f)), // Green channel
      (uchar)(clamp(vel * 128.0f, 0.0f, 255.0f))  // Blue channel
   );
}

__kernel void hsv_velocity(
   __global const float2* velocityIn,
   __global uchar3* rgb_out,
   int width,
   int height
) {
   int x = get_global_id(0);
   int y = get_global_id(1);

   int idx = IDX(x, y, width);
   
   float len = log10( sqrt(exp2(velocityIn[idx].x) + exp2(velocityIn[idx].y)) ); //length of velocity vector
   float angle = atan2(velocityIn[idx].y, velocityIn[idx].x); //angle of velocity vector

   float hue = (angle + M_PI_F) / (2 * M_PI_F); // Normalize angle to [0, 1]
   float sat = clamp(len / 0.5f, 0.0f, 1.0f); // Normalize length to [0, 1]
   float val = 1.0f; // Full brightness

   // Convert HSV to RGB
   float C = val * sat;
   float Ha = hue / 60.0f; // Scale hue to [0, 6]
   float X = C * (1 - fabs(fmod(Ha, 2) - 1));
   float m = val - C;
   float r, g, b;
   if (Ha < 1.0f) {
      r = C; g = X; b = 0;
   } else if (Ha < 2.0f) {
      r = X; g = C; b = 0;
   } else if (Ha < 3.0f) {
      r = 0; g = C; b = X;
   } else if (Ha < 4.0f) {
      r = 0; g = X; b = C;
   } else if (Ha < 5.0f) {
      r = X; g = 0; b = C;
   } else {
      r = C; g = 0; b = X;
   }
   r += m; g += m; b += m;
   // Convert to 0-255 range
   uchar r_out = (uchar)(clamp(r * 255.0f, 0.0f, 255.0f));
   uchar g_out = (uchar)(clamp(g * 255.0f, 0.0f, 255.0f));
   uchar b_out = (uchar)(clamp(b * 255.0f, 0.0f, 255.0f));
   // Store in output
   rgb_out[idx] = (uchar3)(r_out, g_out, b_out);
}
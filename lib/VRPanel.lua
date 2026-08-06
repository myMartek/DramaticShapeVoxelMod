-- VRPanel -- the flat screen as a surface standing in the world.
--
-- OpenXR submits the 2D UI as a composition layer: a quad the runtime draws
-- after the app's own frame, at the runtime's own resolution, sharp and
-- perfectly stable because it is never resampled by our projection.
-- CompositorServices has no equivalent -- no layers at all -- so on visionOS
-- the same picture has to become GEOMETRY, drawn inside the scene like any
-- other object.
--
-- That is a real trade and it is worth naming. As geometry the panel is
-- rasterised by our own camera, so it inherits the eye's resolution and the
-- foveation's density falloff instead of the compositor's crisp layer. What
-- it gains is that it exists in the world: it is occluded by things in front
-- of it, it is stereo by construction, and it needs no second submission path.
--
-- Structurally this is lib/Pokedex.lua with one quad instead of a device: a
-- mesh built once in world pixels, a model matrix from VRRig, and a draw
-- through Voxel3D so it goes down the same pipeline as everything else.
--
-- It draws only what the flat screen is actually saying. The world is already
-- around you; a panel repeating it would be a television in a landscape. So
-- lib/VR.lua asks for it exactly when a menu, a dialog, a battle or a wipe is
-- what the flat screen is showing, and the panel is absent otherwise.

local V = ...

local Mat4 = V.require("Mat4")
local VRRig = V.require("VRRig")
local Voxel3D = V.require("Voxel3D")

local VRPanel = {}


-- Where it hangs and how wide, in the rig's XR metres. Both come from the
-- quad layer these replace, so the panel arrives at the distance and size
-- that was already tuned for it rather than a fresh guess.
VRPanel.FP = { pos = { 0, 0, -1.4 }, width = 1.1 }
VRPanel.DIORAMA = { pos = { 0, 0.1, -1.0 }, width = 0.8 }

-- The GB frame is 160x144, and the panel keeps that shape whatever the
-- window is: the picture on it is the letterboxed frame, not the window.
local ASPECT = 144 / 160

local mesh = nil
local frame = nil        -- { tex, model, u0, v0, u1, v1 } for this eye pass

-- One unit quad, centred on the origin, facing +Z. Everything else -- where it
-- stands, how big it is, which part of the texture it wears -- is the model
-- matrix and the uv, so this is built once and never rebuilt.
local function unitQuad()
  if mesh then return mesh end
  local indices = { 1, 2, 3, 1, 3, 4 }
  local ok, m = pcall(Voxel3D.newMesh, {
    { -0.5, -0.5, 0, 0, 1, 1 },
    {  0.5, -0.5, 0, 1, 1, 1 },
    {  0.5,  0.5, 0, 1, 0, 1 },
    { -0.5,  0.5, 0, 0, 0, 1 },
  }, indices)
  mesh = ok and m or nil
  return mesh
end

-- The letterbox rect, written into the quad's uv. Four vertices, so this is
-- cheaper than keeping a mesh per crop and it follows a window resize for
-- free.
local lastCrop = nil
local function setCrop(u0, v0, u1, v1)
  local key = ("%f,%f,%f,%f"):format(u0, v0, u1, v1)
  if lastCrop == key then return end
  lastCrop = key
  pcall(mesh.setVertices, mesh, {
    { -0.5, -0.5, 0, u0, v1, 1 },
    {  0.5, -0.5, 0, u1, v1, 1 },
    {  0.5,  0.5, 0, u1, v0, 1 },
    { -0.5,  0.5, 0, u0, v0, 1 },
  })
end

-- Put the panel where the head is looking, for this frame.
--
--   tex          the flat frame
--   u0..v1       the letterbox rect inside it, 0..1
--   pose         the head pose this frame (VRXR's shape)
--   pivot/anchor/scale/yaw   the same mapping the eyes and the pokedex use,
--                            so the panel sits in the world the player is in
--                            rather than in a space of its own
--   fp           first person, which wants it further out and wider
function VRPanel.place(tex, u0, v0, u1, v1, pose, pivot, anchor, scale, yaw, fp)
  if not (tex and pose) then VRPanel.clear() return end

  -- CLAMP the wrap mode, once per texture.
  --
  -- This is the virtual screen: a texture LOVE created for its own backbuffer,
  -- adopted here, and nothing ever set its wrap. LOVE's default is "repeat",
  -- so the quad's edges sample past 0..1 and tile the picture -- the world
  -- beside the menu box appeared ten or twenty times over. The rest of the
  -- scene never noticed because a tileset atlas is addressed well inside its
  -- own bounds.
  --
  -- Filter too: this is pixel art shown large, and the default bilinear blur
  -- is not what a Game Boy frame should look like at a metre's distance.
  if VRPanel._wrapped ~= tex then
    VRPanel._wrapped = tex
    pcall(tex.setWrap, tex, "clamp", "clamp")
    pcall(tex.setFilter, tex, "nearest", "nearest")
  end
  if not unitQuad() then VRPanel.clear() return end

  local spec = fp and VRPanel.FP or VRPanel.DIORAMA

  -- The panel hangs in FRONT OF THE HEAD, upright: a surface that rolled and
  -- pitched with the head would be unreadable.
  --
  -- The direction comes straight out of the pose's own rotation rather than
  -- through VRRig.headYawPitch. That function answers in the MOD's compass,
  -- where increasing yaw turns left, and using it to rotate an offset that is
  -- expressed in XR axes mixed two conventions -- the panel came out behind
  -- the camera.
  local q = pose.quat
  local R = Mat4.fromQuat(q[1], q[2], q[3], q[4])
  -- R's third column is the head's +Z; forward is its negation. Flattened,
  -- because only the heading should carry over.
  local fx, fz = -R[3], -R[11]
  local flen = math.sqrt(fx * fx + fz * fz)
  if flen < 1e-6 then fx, fz, flen = 0, -1, 1 end
  fx, fz = fx / flen, fz / flen
  -- Ry(a) sends (0,0,-1) to (-sin a, 0, -cos a), which solves to this.
  local yawOnly = math.atan2(-fx, -fz)
  local ox, oy, oz = spec.pos[1], spec.pos[2], spec.pos[3]
  -- The offset, turned to face the same way: right is perpendicular to
  -- forward in the ground plane.
  local rx, rz = -fz, fx
  local px = pose.pos[1] + rx * ox + fx * (-oz)
  local pz = pose.pos[3] + rz * ox + fz * (-oz)
  local py = pose.pos[2] + oy

  local flat = {
    pos = { px, py, pz },
    -- Facing the head, upright: a yaw about +Y and nothing else.
    quat = { 0, math.sin(yawOnly * 0.5), 0, math.cos(yawOnly * 0.5) },
  }
  local model = VRRig.propMatrix(flat, pivot, anchor, scale, yaw)

  -- The quad is a unit square, so its size is the model's -- in METRES,
  -- because propMatrix already carries the metres-to-world-pixels scale.
  -- Multiplying by it again here made the panel ten times too wide.
  local w = spec.width
  model = Mat4.mul(model, Mat4.scale(w, w * ASPECT, 1))

  setCrop(u0 or 0, v0 or 0, u1 or 1, v1 or 1)
  frame = { tex = tex, model = model }

end

function VRPanel.clear()
  frame = nil
end

function VRPanel.active()
  return frame ~= nil
end

-- Drawn from the eye pass, alongside the pokedex and the gun, so it is one
-- more object in the scene rather than a stage of its own.
function VRPanel.draw()
  local f = frame
  if not (f and mesh) then
    if (VRPanel._t or 0) % 60 == 1 then print("[panel] draw: nothing placed") end
    return
  end
  local ok, err = pcall(Voxel3D.draw, mesh, f.tex, f.model)
  if not ok and (VRPanel._d or 0) < 1 then
    VRPanel._d = 1
    print("[panel] draw failed: " .. tostring(err))
  end
end

return VRPanel

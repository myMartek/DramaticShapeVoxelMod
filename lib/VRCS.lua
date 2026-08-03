-- VRCS -- the visionOS backend, sitting where VRXR sits on Windows.
--
-- VRXR reaches OpenXR through LuaJIT's FFI and blits each eye canvas into a
-- swapchain image with raw GL.  There is none of that here: love.xr is a real
-- module inside LOVE, and an eye is a love Canvas the renderer draws straight
-- into.  So this file is small, and deliberately so -- it is a translation
-- layer, not a second implementation.
--
-- It mirrors VRXR's public surface exactly, because lib/VR.lua is written
-- against that surface and should not have to know which one it is talking to.
-- The shapes it returns are VRXR's shapes: pose.pos and pose.quat are ARRAYS,
-- the field of view is four SIGNED angles in OpenXR's convention.  VRRig and
-- Mat4.fovProjection then work unchanged, which is the entire point.
--
-- Same failure discipline as VRXR (lib/VRXR.lua:19-22): every entry point is
-- pcall-wrapped, failure is a status string, and nothing here ever raises into
-- the render pipeline.

local VRCS = {}

VRCS.kind = "visionos"

-- CompositorServices paces the app through beginFrame, exactly as xrWaitFrame
-- does on OpenXR.
VRCS.takesOverVSync = true

-- No mirror window: the host has no window of its own here, and the flat frame
-- already lives in the virtual screen.
VRCS.hasMirror = false

-- CompositorServices has no composition layers at all -- no equivalent of
-- XrCompositionLayerQuad.  The 2D UI has to become geometry in the scene.
VRCS.hasQuadLayer = false

local status = "not started"
local views = nil

function VRCS.status() return status end

local function available()
  return love.xr ~= nil and love.xr.available()
end

VRCS.available = available

-- Takes the frame loop from the host shell and brings world tracking up.
function VRCS.start()
  if not available() then
    status = "love.xr not available (not a visionOS build?)"
    return false
  end
  local ok, claimed = pcall(love.xr.claim)
  if not ok or not claimed then
    status = "compositor did not hand over the frame loop"
    return false
  end
  status = "session created"
  return true
end

function VRCS.stop()
  pcall(function() if love.xr then love.xr.release() end end)
  views = nil
  status = "stopped"
end

-- OpenXR needs its event queue pumped to notice a session ending.  Here the
-- layer's state is the whole story, so this is a state read.
function VRCS.poll()
  if not available() then return false end
  local ok, s = pcall(love.xr.state)
  if not ok then return false end
  return s ~= "invalidated"
end

function VRCS.isRunning()
  if not available() then return false end
  local ok, s = pcall(love.xr.state)
  return ok and s == "running"
end

-- Blocks until the compositor wants a frame.  Returns the predicted display
-- time and whether to render, matching VRXR.waitFrame's contract; the time is
-- opaque to callers, which only pass it back in.
function VRCS.waitFrame()
  if not available() then return nil, false end
  local ok, time = pcall(love.xr.beginFrame)
  if not ok or time == nil then
    views = nil
    return nil, false
  end
  local okv, v = pcall(love.xr.views)
  views = (okv and v) or nil
  return time, views ~= nil
end

-- One camera per eye.  Already in VRXR's shape, so this is a pass-through --
-- the conversion happens in C, where the projection matrix is, rather than
-- being reconstructed here from tangents.
function VRCS.locateViews()
  return views
end

-- The drawable's texture for this eye, as a Canvas the scene renders into.
-- VRXR hands back a GL texture name that VRGL then blits a canvas into; there
-- is no blit here at all, which is a frame's worth of copying saved per eye.
function VRCS.eyeCanvas(i)
  if not available() then return nil end
  local ok, canvas = pcall(love.xr.eyeCanvas, i)
  if not ok then return nil end
  return canvas
end

-- VRXR acquires and releases swapchain images around each eye.  Compositor
-- drawables have no such handshake, so these exist only so lib/VR.lua can call
-- the same sequence for either backend.
function VRCS.acquireEye(i)
  local canvas = VRCS.eyeCanvas(i)
  if not canvas then return nil end
  return canvas, canvas:getWidth(), canvas:getHeight()
end

function VRCS.releaseEye() end

function VRCS.endFrame()
  if not available() then return false end
  local ok, done = pcall(love.xr.submitFrame)
  return ok and done == true
end

-- Input is not wired yet: the pad reaches the game through SDL's own path, and
-- the PSVR2 Sense controllers need ARKit accessory tracking.  Returning nil is
-- a state lib/VR.lua already handles -- it is what OpenXR's simple_controller
-- profile produces, which has no poses either.
function VRCS.input()
  return nil
end

-- OpenXR only: a floating quad layer for the flat UI.  Callers check
-- hasQuadLayer before using these.
function VRCS.quadSize() return 0, 0 end
function VRCS.acquireQuad() return nil end
function VRCS.releaseQuad() end

return VRCS

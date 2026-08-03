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

-- Declared here because stop() below is compiled before the eye-buffer
-- section defines it, and a local that is not yet in scope would resolve to a
-- nil global instead -- an error that only fires when someone leaves VR.
local dropScratch

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
  -- Two eye-sized canvases is tens of megabytes; leaving them allocated
  -- across a session the player has left is the kind of thing that only
  -- shows up as a memory warning much later, in some unrelated scene.
  dropScratch()
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

-- ------- the eye buffers, and the one flip between them
--
-- The scene does NOT draw straight into the compositor's texture, even though
-- it could.  It draws into a scratch canvas of our own and releaseEye copies
-- that across, turned upside down.  One full-screen textured quad per eye, so
-- roughly a third of a millisecond and two canvases' worth of memory.  That
-- is a real cost and it is worth stating why it is paid.
--
-- lib/Voxel3D.lua premultiplies Mat4.scale(1,-1,1) onto every projection,
-- because the mod bypasses LOVE's transform_projection and has to reproduce
-- the Y inversion LOVE applies to canvas projections itself.  On OpenGL that
-- is exactly right.  On Metal LOVE inverts nothing -- its render targets are
-- top-left already -- so the world canvas comes out vertically mirrored, and
-- the host quietly turns it back over when it composites (see the
-- worldOverride blit in src/render/Renderer.lua: it draws with a negative Y
-- scale on iOS under LOVE 12, and only there).  Two errors that cancel.
--
-- The flip is not confined to that one matrix, though.  It is a CONVENTION:
-- "after vp, clip Y maps to the canvas row as y * 0.5 + 0.5".  Voxel3D's
-- horizonY, horizonLine, skyBody, drawWorldDisc and project all read it that
-- way, and so do the water shaders.  Removing the flip for the eyes alone
-- would silently invert all of them, on a path that can only be checked by
-- putting the headset on.
--
-- So the convention stays whole and the correction happens once, here, where
-- the image leaves the mod -- the same correction the host already makes for
-- the flat screen, in the same place in the pipeline.  When the convention is
-- eventually unified against love.graphics.getRendererInfo() (the honest
-- fix, and a much wider change), this scratch canvas and its blit are what
-- gets deleted.

local scratch = {}

local function eyeScratch(i, model)
  local w, h = model:getWidth(), model:getHeight()
  local c = scratch[i]
  if c and c:getWidth() == w and c:getHeight() == h then return c end
  if c then pcall(c.release, c) end
  -- Same pixel format as the drawable, so the copy is a copy and not a
  -- silent colour-space conversion.
  local ok, made = pcall(love.graphics.newCanvas, w, h,
                         { format = model:getFormat() })
  if not ok then
    -- A driver that will not give us that format is not a reason to lose the
    -- frame: fall back to the default and accept whatever conversion the
    -- blit then does.
    ok, made = pcall(love.graphics.newCanvas, w, h)
  end
  scratch[i] = ok and made or nil
  return scratch[i]
end

-- Assigns the forward declaration at the top of the file.
function dropScratch()
  for i, c in pairs(scratch) do
    pcall(c.release, c)
    scratch[i] = nil
  end
end

-- The canvas this eye's scene renders into.  Not the compositor's texture
-- itself -- see above.
function VRCS.eyeCanvas(i)
  if not available() then return nil end
  local ok, canvas = pcall(love.xr.eyeCanvas, i)
  if not ok or not canvas then return nil end
  return eyeScratch(i, canvas) or canvas
end

-- VRXR acquires and releases swapchain images around each eye.  Compositor
-- drawables have no such handshake, so acquireEye exists only so lib/VR.lua
-- can call the same sequence for either backend.  releaseEye, by contrast,
-- does real work here: it is where the eye actually reaches the compositor.
function VRCS.acquireEye(i)
  local canvas = VRCS.eyeCanvas(i)
  if not canvas then return nil end
  return canvas, canvas:getWidth(), canvas:getHeight()
end

function VRCS.releaseEye(i)
  local src = scratch[i]
  if not src or not available() then return end
  local ok, dst = pcall(love.xr.eyeCanvas, i)
  if not ok or not dst or dst == src then return end
  pcall(function()
    love.graphics.setCanvas(dst)
    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setBlendMode("replace", "premultiplied")
    love.graphics.setColor(1, 1, 1, 1)
    -- Drawn from the bottom edge upward: this is the vertical mirror that
    -- undoes Voxel3D's clip-space flip, and it is the whole point of the
    -- scratch canvas.
    love.graphics.draw(src, 0, src:getHeight(), 0, 1, -1)
    love.graphics.setBlendMode("alpha")
    love.graphics.setCanvas()
  end)
end

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

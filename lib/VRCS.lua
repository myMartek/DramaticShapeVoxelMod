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

-- The session can come back by itself, and routinely does.
--
-- The compositor layer exists only while the immersive space is open, and
-- the player closes and reopens that with the Digital Crown as a matter of
-- course -- a new layer arrives each time. Unlike an OpenXR runtime going
-- away, which really is "something broke, tell the player", this is an
-- ordinary thing that happens several times a session, so lib/VR.lua must
-- keep retrying instead of latching the failure.
VRCS.resumable = true

local status = "not started"
local views = nil

-- Declared here because stop() below is compiled before the eye-buffer
-- section defines it, and a local that is not yet in scope would resolve to a
-- nil global instead -- an error that only fires when someone leaves VR.
local dropScratch

function VRCS.status() return status end

-- Whether a frame can be driven THIS INSTANT: a compositor layer is attached.
-- False in the launcher window, true inside the immersive space. Every
-- per-frame entry point below guards on this.
local function available()
  return love.xr ~= nil and love.xr.available()
end

VRCS.available = available

-- Whether this build can do VR at all -- which is what decides whether the mod
-- OFFERS its VR row, and is a different question entirely.
--
-- The distinction is not academic. main.lua builds the settings schema once,
-- when the mod loads, and that happens at app start in the launcher window
-- where no immersive space exists yet. Asking `available` there answers "no
-- layer attached", the VR row is left out of the schema, and since the schema
-- is built exactly once, entering the space later cannot put it back. The row
-- the player is supposed to switch VR on with simply was not there.
--
-- love.xr exists only in a visionOS build, so its presence is nearly the whole
-- answer; love.xr.supported() is the engine agreeing. The fallback covers an
-- engine build from before that function existed.
function VRCS.supported()
  if love.xr == nil then return false end
  if love.xr.supported ~= nil then
    local ok, yes = pcall(love.xr.supported)
    if ok then return yes == true end
  end
  return true
end

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

-- Assigns the forward declaration at the top of the file.
function dropScratch()
  -- The native bridge owns these intermediary textures; Lua only forgets its
  -- frame-local references when the compositor layer goes away.
  for i in pairs(scratch) do scratch[i] = nil end
end

-- A foveated intermediary: it reports logical screen dimensions to LÖVE, but
-- physically contains only the fragments selected by the flipped gaze map.
function VRCS.eyeCanvas(i)
  if not available() then return nil end
  local ok, canvas = pcall(love.xr.eyeCanvas, i)
  if not ok or not canvas then return nil end
  scratch[i] = canvas
  return canvas
end

function VRCS.eyeDepth(i)
  if not available() or not love.xr.eyeDepth then return nil end
  local ok, depth = pcall(love.xr.eyeDepth, i)
  return ok and depth or nil
end

-- A conventional, de-foveated snapshot of this eye: normal textures with
-- normal UVs, which is what a SCREEN-SPACE pass needs.
--
-- The packed attachments cannot be sampled by such a pass at all -- once
-- variable rasterization is on, a logical UV is not a physical texture UV --
-- and approximating the conversion through a 256x256 lookup was close enough
-- near the gaze point and wrong everywhere else. Which is why neither
-- orientation of that lookup made the water's occlusion correct: the
-- approximation was the fault, not its direction.
function VRCS.resolveEye(i)
  if not available() or not love.xr.resolveEye then return nil end
  local ok, colour, depth = pcall(love.xr.resolveEye, i)
  if not ok then return nil end
  return colour, depth
end

-- The physical size of this eye's packed attachments, which is the only thing
-- a screen-space pass needs to address them: its fragment coordinate is
-- already physical, so dividing by this gives the right UV directly and no
-- rate-map conversion is involved at all.
-- The flat frame, as a texture.
--
-- LOVE renders headless here, so the 2D UI is already sitting in an offscreen
-- "virtual screen" -- no front-buffer read, no copy. VRXR has to go through
-- raw GL for the same picture because on Windows it lives in a window.
function VRCS.screenTexture()
  if not available() or not love.xr.screenTexture then return nil end
  local ok, tex = pcall(love.xr.screenTexture)
  if not ok then return nil end
  return tex
end

function VRCS.eyePhysicalSize(i)
  if not available() or not love.xr.eyePhysicalSize then return nil end
  local ok, w, h = pcall(love.xr.eyePhysicalSize, i)
  if not ok or not w then return nil end
  return w, h
end

function VRCS.rateMap(i)
  if not available() or not love.xr.rateMap then return nil end
  local ok, map = pcall(love.xr.rateMap, i)
  return ok and map or nil
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
  if love.xr.submitEye then
    pcall(love.xr.submitEye, i, src)
  end
end

function VRCS.endFrame()
  if not available() then return false end
  local ok, done = pcall(love.xr.submitFrame)
  return ok and done == true
end

-- ------- input
--
-- STICKS ONLY, deliberately.
--
-- The pad already reaches the game: SDL enumerates it, LOVE delivers its
-- events, and the engine's own handler walks and presses buttons with it.
-- That is why the left stick and A/B work with no help from here.
--
-- What does NOT work that way is anything the MOD owns rather than the engine
-- -- above all the right stick, which is this mod's snap/smooth turn and its
-- diorama zoom. Those are read from the ctl table alone, so returning nil
-- meant lib/VR.lua's driveControls bailed out on its first line and the right
-- stick did nothing at all.
--
-- Buttons are left nil ON PURPOSE. driveControls feeds them through setGB,
-- which is edge-triggered on its own overlay channel; reporting a button the
-- engine is already delivering would press it twice. Left nil, setGB's
-- `down` is falsy and its `held` entry never set, so it neither presses nor
-- releases and the working path is untouched. When the PSVR2 Sense pair
-- arrives -- whose buttons the engine has no path for -- this is where they
-- go, and the double-press question has to be answered then.
--
-- Poses (handl/handr/aimr) stay absent: they need ARKit accessory tracking.
-- That is a state lib/VR.lua already handles, being what OpenXR's
-- simple_controller profile produces.

local DEADZONE = 0.12

local function axis(pad, name)
  local ok, v = pcall(pad.getGamepadAxis, pad, name)
  if not ok or type(v) ~= "number" then return 0 end
  -- Zeroed, not rescaled: driveControls applies its own thresholds (0.2 for
  -- the smooth turn, 0.65/0.35 for the snap's hysteresis, 0.15 for the zoom)
  -- and rescales from them. Rescaling here too would move all three.
  if math.abs(v) < DEADZONE then return 0 end
  return v
end

local function firstGamepad()
  local ok, pads = pcall(love.joystick.getJoysticks)
  if not ok or type(pads) ~= "table" then return nil end
  for _, pad in ipairs(pads) do
    local okg, isPad = pcall(pad.isGamepad, pad)
    if okg and isPad then return pad end
  end
  return nil
end

-- ------- hands
--
-- A PINCH ANCHORS A STICK. The moment thumb and finger meet, that point in
-- space becomes the stick's centre; moving the hand away from it deflects the
-- stick, and letting go recentres it. It reads like holding something, it
-- needs no calibration, and it costs the player nothing to learn -- which
-- matters, because there is no way to label a gesture on screen.
--
--   LEFT  index   walk        (the deflection is the direction and the speed)
--   LEFT  middle  turn        (sideways only; this mod turns in snaps or a rate)
--   RIGHT index   A
--   RIGHT middle  B
--   either fist   START
--
-- Buttons are edges, not levels: a pinch held while a menu opens must not keep
-- pressing A. VRCS reports current and changed, and lib/VR.lua's driveControls
-- already reads that pair.
local HAND_STICK = 0.12   -- metres of travel for full deflection

local held = { }          -- per gesture: the anchor point, while it is held
local was = { }           -- last frame's state, for the edges

local function stick(key, hand, on)
  if not on then held[key] = nil return 0, 0 end
  local p = hand.pose and hand.pose.pos
  if not p then return 0, 0 end
  if not held[key] then held[key] = { p[1], p[2], p[3] } return 0, 0 end
  local a = held[key]
  local dx = (p[1] - a[1]) / HAND_STICK
  local dz = (p[3] - a[3]) / HAND_STICK
  -- Forward is -Z, and the ctl table wants +Y forward.
  local x = math.max(-1, math.min(1, dx))
  local y = math.max(-1, math.min(1, -dz))
  return x, y
end

local function edge(key, on)
  local changed = (was[key] or false) ~= on
  was[key] = on
  return on, changed
end

-- Deliberately NOT gated on available(). A compositor layer means "VR can
-- present a frame", which is a different question from "the player's hands are
-- being tracked" -- and tying the two meant the gestures were dead everywhere
-- the flat screen is shown: the launcher window, and the space before the VR
-- row is switched on. love.xr.hands starts tracking by itself now.
local function handInput()
  if love.xr == nil or not love.xr.hands then return nil end
  local ok, hands = pcall(love.xr.hands)
  if not ok or type(hands) ~= "table" then return nil end
  local L, R = hands[1], hands[2]


  if not ((L and L.tracked) or (R and R.tracked)) then return nil end

  local ctl = {}

  if L and L.tracked then
    local mx, my = stick("walk", L, L.pinchIndex)
    ctl.moveX, ctl.moveY = mx, my
    -- Turning is sideways only: the vertical half of that gesture would
    -- fight the zoom, and there is nothing else it should mean.
    local tx = stick("turn", L, L.pinchMiddle)
    ctl.lookX, ctl.lookY = tx, 0
    ctl.handl = L.pose
  else
    held.walk, held.turn = nil, nil
  end

  if R and R.tracked then
    ctl.a, ctl.aChanged = edge("a", R.pinchIndex == true)
    ctl.b, ctl.bChanged = edge("b", R.pinchMiddle == true)
    ctl.handr = R.pose
    ctl.aimr = R.pose
  else
    ctl.a, ctl.aChanged = edge("a", false)
    ctl.b, ctl.bChanged = edge("b", false)
  end

  -- A fist from either hand. Two hands can make one at once and that is still
  -- one press, not two.
  local fist = (L and L.tracked and L.fist) or (R and R.tracked and R.fist) or false
  ctl.start, ctl.startChanged = edge("start", fist == true)

  return ctl
end

-- Also published on its own, because the flat screen wants the gestures
-- WITHOUT the pad fallback underneath them: outside VR the pad already reaches
-- the engine through LOVE's own joystick path, and answering it a second time
-- here would drive every axis twice.
VRCS.handInput = handInput

function VRCS.input()
  -- Hands first where they are tracked: someone who has put the pad down and
  -- raised their hands means the hands.
  local hands = handInput()
  if hands then return hands end

  local pad = firstGamepad()
  if not pad then return nil end

  -- SDL's Y axes run +DOWN. The ctl table is in OpenXR's convention, +UP,
  -- because that is what driveControls was written against -- it negates
  -- moveY again on the way to the engine's lefty, and reads a positive lookY
  -- as "zoom in". Getting this sign wrong inverts walking and zooming without
  -- breaking anything loudly enough to notice.
  return {
    moveX =  axis(pad, "leftx"),
    moveY = -axis(pad, "lefty"),
    lookX =  axis(pad, "rightx"),
    lookY = -axis(pad, "righty"),
  }
end

-- OpenXR only: a floating quad layer for the flat UI.  Callers check
-- hasQuadLayer before using these.
function VRCS.quadSize() return 0, 0 end
function VRCS.acquireQuad() return nil end
function VRCS.releaseQuad() end

return VRCS

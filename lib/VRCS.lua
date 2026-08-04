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

local function eyeScratch(i, model)
  local w, h = model:getWidth(), model:getHeight()
  local c = scratch[i]
  if c and c:getWidth() == w and c:getHeight() == h then return c end
  if c then pcall(c.release, c) end
  -- dpiscale = 1 is NOT optional, and leaving it out is what made the eye a
  -- portrait patch in a black field.
  --
  -- LOVE sizes a canvas in UNITS and multiplies by the dpi scale to get
  -- pixels, so newCanvas(2048, 1984) on a display that reports a scale of 2
  -- is 4096x3968 pixels while still answering 2048x1984 to getWidth(). The
  -- scene then fills those pixels, and the 1:1 blit into a 2048-wide drawable
  -- copies a corner of it. lib/PixelCanvas.lua pins the same value for the
  -- same reason; this is a compositor texture measured in real pixels, and
  -- units have to mean pixels here.
  --
  -- Same pixel format as the drawable too, so the copy is a copy and not a
  -- silent colour-space conversion.
  local ok, made = pcall(love.graphics.newCanvas, w, h,
                         { format = model:getFormat(), dpiscale = 1 })
  if not ok then
    -- A driver that will not give us that format is not a reason to lose the
    -- frame: fall back to the default and accept whatever conversion the
    -- blit then does. dpiscale stays pinned.
    ok, made = pcall(love.graphics.newCanvas, w, h, { dpiscale = 1 })
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

-- TEMPORARY (visionOS port): the eye map is taken on a SETTLED frame.
--
-- The first reading showed the top 45% of the canvas drawn and the rest
-- black, which looks like a broken vertical projection -- but it was frame
-- one, where the ground mesh does not exist yet, and "the terrain has not
-- been built" looks exactly the same from a colour sample. Nothing can be
-- concluded from that. This waits until the world has had a few seconds to
-- finish meshing, so a black lower half means the projection and not the
-- chunk queue.
local releaseFrames = 0
local MAP_AT_FRAME = 400

function VRCS.releaseEye(i)
  local src = scratch[i]
  local ok, dst = pcall(love.xr.eyeCanvas, i)

  releaseFrames = releaseFrames + 1
  local loggedRelease = releaseFrames ~= MAP_AT_FRAME

  -- TEMPORARY (visionOS port): whether this path runs at all.
  -- Every early return below is silent, and a silent skip looks exactly like
  -- a blit that misses -- so "no green appeared" would say nothing.
  if not loggedRelease then
    loggedRelease = true
    local function dims(o)
      if not o then return "nil" end
      local w, h, pw, ph = -1, -1, -1, -1
      pcall(function()
        w, h = o:getWidth(), o:getHeight()
        pw, ph = o:getPixelWidth(), o:getPixelHeight()
      end)
      return ("%dx%d units / %dx%d px"):format(w, h, pw, ph)
    end
    print(("[vr-probe] releaseEye: available=%s scratch=%s drawable=%s same=%s")
      :format(tostring(available()), dims(src), dims(ok and dst or nil),
              tostring(src ~= nil and dst == src)))

    -- The frustum this eye was given, and the shape it implies.
    --
    -- A vertical field of view that disagrees with the texture is the
    -- remaining explanation for a horizontal cut, and it is one number away
    -- from being settled: the tangent extents of the frustum, as an aspect,
    -- must match the texture's. If they do, the projection is innocent and
    -- the black half is content that was never drawn.
    local v = views and views[1]
    if v and v.fov then
      local f = v.fov
      local tw = math.tan(f.angleRight) - math.tan(f.angleLeft)
      local th = math.tan(f.angleUp) - math.tan(f.angleDown)
      print(("[vr-probe] fov L=%.4f R=%.4f U=%.4f D=%.4f | tan extent %.4f x %.4f")
        :format(f.angleLeft, f.angleRight, f.angleUp, f.angleDown, tw, th))
      print(("[vr-probe] frustum aspect=%.4f texture aspect=%.4f")
        :format(tw / (th ~= 0 and th or 1), (v.w or 1) / (v.h or 1)))
    end

    -- WHERE the picture actually is, read off the eye itself.
    --
    -- Three explanations for the portrait patch have been measured and
    -- disproved, and asking someone in a headset to describe a border is a
    -- slow and lossy way to find a fourth. This reads the canvas back and
    -- prints a coarse map of it: '#' where a pixel carries colour, '.' where
    -- it is black. The shape of the drawn region is then a fact rather than a
    -- description, and its edges say whether they fall at the flat view's
    -- 9:16 or somewhere else entirely.
    if src then
      local okR, img = pcall(love.graphics.readbackTexture, src)
      if okR and img then
        local W, H = src:getPixelWidth(), src:getPixelHeight()
        local COLS, ROWS = 24, 16
        print(("[vr-probe] eye map %dx%d (# = drawn, . = black):"):format(W, H))
        for r = 0, ROWS - 1 do
          local line = {}
          for c = 0, COLS - 1 do
            local x = math.min(W - 1, math.floor((c + 0.5) * W / COLS))
            local y = math.min(H - 1, math.floor((r + 0.5) * H / ROWS))
            local okP, pr, pg, pb = pcall(img.getPixel, img, x, y)
            local lit = okP and (pr + pg + pb) > 0.02
            line[#line + 1] = lit and "#" or "."
          end
          print("[vr-probe] |" .. table.concat(line) .. "|")
        end
      else
        print("[vr-probe] eye map: readback failed")
      end
    end
  end

  if not src or not available() then return end
  if not ok or not dst or dst == src then return end
  pcall(function()
    love.graphics.setCanvas(dst)
    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setBlendMode("replace", "premultiplied")
    -- TEMPORARY (visionOS port): the eye picture fills only a portrait patch
    -- of the view, and three separate explanations for that have now been
    -- measured and disproved (render size, world extent, dpi scale -- the
    -- canvas is 2048x1984 units AND pixels, exactly the drawable). So mark
    -- the parts of the compositor texture this blit does NOT cover.
    --
    -- The green clear that used to be here is gone: no green ever showed,
    -- which means this blit covers the drawable completely and the black
    -- borders are inside the SOURCE. The eye map above measures that
    -- directly, and a green fill would have made every sample read as drawn.
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

function VRCS.input()
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

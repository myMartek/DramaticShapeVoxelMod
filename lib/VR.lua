-- VR: the conductor -- one call per game frame that runs the whole
-- headset side, and the row that switches it on.
--
-- The shape of a VR frame, from the pipeline's update hook (which ticks
-- every frame whatever is on the stack, which is exactly what a headset
-- needs -- the world must keep arriving through menus, dialogs and
-- battles):
--
--   poll the runtime's events (begin the session when it says READY)
--   xrWaitFrame            <- BLOCKS until the headset wants a frame;
--                             with vsync handed off (set to 0 while the
--                             session runs) this is what paces the whole
--                             app at headset rate, while FixedStep keeps
--                             the game's own logic at its 60 Hz
--   locate the two eyes
--   render the world once per eye (VoxelScene.render's `eyes` path:
--     shared shadow map, shared pose capture, per-eye cameras from VRRig)
--   blit each eye canvas into its swapchain image (VRGL)
--   copy the window's front buffer into the UI quad when a menu, dialog,
--     battle or wipe is what the flat screen is showing
--   xrEndFrame with the projection layer and/or the quad
--
-- WHICH VR YOU GET mirrors the VOXEL ladder, deliberately: on the orbit
-- rungs the world is a TABLETOP DIORAMA pinned below and ahead of where
-- your head started -- lean in, walk around it; on 1ST you stand inside
-- at life scale, the HMD steers FirstPerson's yaw and pitch, and FreeMove
-- walks where you look exactly as it does on the flat screen. A STAGED
-- FIGHT takes the camera from both: the headset snaps -- through a fade
-- to black and back -- to the flat battle's own over-the-shoulder seat
-- (VRRig.battleMount), and returns the same way when the fight ends;
-- the 2D battle screen lights up on the POKEDEX in the tracked left
-- hand (lib/Pokedex.lua) and NO floating panel is submitted at all --
-- the fight itself owns the view. The flat window keeps
-- running as the mirror (left eye when the world is up), so menus stay
-- usable at the desk and every existing input keeps working alongside
-- the XR controllers.
--
-- Failure is a status, never a crash: no runtime, no headset, no GL
-- interop, or a mid-session loss all land back on the flat screen with
-- the reason readable off VR.status().

-- the mod namespace (see main.lua): V.require loads a sibling module
local V = ...

local ModSetting = V.require("ModSetting")
local Voxel = V.require("VoxelState")
local Voxel3D = V.require("Voxel3D")
local VoxelScene = V.require("VoxelScene")
local FirstPerson = V.require("FirstPerson")
local BattleCam = V.require("BattleCam")
local Mat4 = V.require("Mat4")
local VRRig = V.require("VRRig")
-- Deliberately still called VRXR: every other reference in this file then
-- needs no change at all.  What it now holds is whichever backend the
-- platform has -- OpenXR on Windows, CompositorServices on visionOS.
local VRBackend = V.require("VRBackend")
-- Empty table rather than nil when no backend applies: VR.update never runs
-- then (VR.supported() is false), but VR.mirror and VR.invalidate are called
-- unconditionally from main.lua and must not fault on a nil index.
local VRXR = VRBackend.get() or {}
-- VRGL is raw WGL and only ever loads alongside OpenXR.  On visionOS an eye is
-- a Canvas the scene draws into, so there is nothing to blit and nothing to
-- load.
local VRGL = (VRBackend.kind() == "openxr") and V.require("VRGL") or nil
local Pokedex = V.require("Pokedex")

local VR = {}

-- the row: plain OFF/ON. No hotkey -- the engine's display keys are
-- spoken for, and a headset is not something to toggle by accident.
VR.setting = ModSetting.new("vr", "VR", { false, true }, { "OFF", "ON" })

-- How the right stick turns you in first person. OFF is the 45-degree
-- SNAP this mod shipped with and the reason for it is comfort, not
-- taste: a software turn moves the world past a head that did not move,
-- which is vection with no vestibular signal to match it, and it is the
-- single most reliable way to make somebody ill in a headset. A snap
-- gives the inner ear nothing to disagree with.
--
-- But snap turning is not free either -- it costs continuity, and the
-- players who have their sea legs generally want the stick. So it is a
-- row rather than a decision: OFF by default, on for anyone who asks,
-- and the row only exists while there is a headset to use it in.
VR.smoothTurn = ModSetting.new("smoothturn", "SMOOTH TURN",
                               { false, true }, { "OFF", "ON" })

-- radians per second at full deflection, with a squared response so the
-- first half of the throw aims and the rest turns -- the same curve
-- FirstPerson gives the flat screen's right stick
VR.SMOOTH_TURN_RATE = 2.2

-- Where the diorama's UI panel floats vs first person's. These are the
-- FALLBACK screens: wherever the pokedex is up and lit -- first
-- person's menus, a battle's 2D scene -- no quad is submitted at all
-- (see updateQuad), and these serve only the diorama and the no-tracked-
-- controller case.
local QUAD_DIORAMA = { pos = { 0, 0.1, -1.0 }, width = 0.8 }
local QUAD_FP = { pos = { 0, 0, -1.4 }, width = 1.1 }

local started = false           -- start() succeeded this enablement
-- last "not yet" reported by a resumable backend, so the wait is logged
-- once rather than every frame
local waiting = nil
local failed = nil              -- start() failed; wait for a re-toggle
local wasOn = false
local savedVsync = nil
local fboCache = setmetatable({}, { __mode = "k" })   -- canvas -> GL FBO id
local mirrorSrc = nil           -- last left-eye canvas, for the window
local mirrorCanvas = nil
local status = "off"

-- the diorama's live adjustments: the right stick's zoom (a multiplier on
-- the model's size) and the grab-drag's height (metres of world travel)
local zoom = 1
local heightOff = 0
local held = {}                 -- GB buttons this module is holding down
local lastHandY = nil           -- the gripping hand's height, last frame

-- First person's SNAP TURN: the right stick flicked left or right steps
-- the whole XR-to-world mapping 45 degrees at a time (a smooth software
-- turn is the classic comfort mistake -- vection with no vestibular
-- signal; a snap is instant and the head does the rest). The offset
-- turns the mapping itself, so the eyes, the walk direction and the
-- pokedex all agree about which way the world now faces.
local SNAP_TURN = math.rad(45)
local fpYawOff = 0              -- accumulated snaps, radians
local snapArmed = true          -- re-arms when the stick returns to centre

local function wrapPi(a)
  return (a + math.pi) % (2 * math.pi) - math.pi
end

-- The battle snap, made a FADE rather than a cut: when a fight is staged
-- on the world (or stops being), black rises over both eyes, the camera
-- swaps mounts behind it, and black lifts. A teleport inside VR is the
-- one camera move that should never be SEEN happening -- the world
-- sliding to a new seat reads as the room moving.
local FADE_TIME = 0.35          -- seconds each way: out, then back in
local camMode = "explore"       -- "explore" (diorama / 1ST) or "battle"
local fadeAlpha = 0             -- the black over the eyes right now

-- The staged fight to look at, if there is one: arena, floor height.
local function battleStage()
  local ok, arena, groundY = pcall(function()
    return V.require("OverworldBattle").stage()
  end)
  if not ok then return nil end
  return arena, groundY
end

-- the palette closure the engine hands drawWorld; stashed there (see
-- main.lua) because the VR frame renders from update, where no ctx exists
VR.paletteFor = nil

-- Whether this platform can do VR AT ALL: the shipped loader and the GL
-- interop are Win32 (openxr_loader.dll, wgl), so only Windows qualifies.
-- Everywhere else -- Android above all -- the row is not offered on any
-- menu, and a stored vr=true is ignored rather than read: a save that
-- migrated over from the desktop must not leave a phone trying to start
-- an OpenXR session (or silently forcing the battle rows). Headless runs
-- have no love.system and answer true, which costs nothing: enabling VR
-- there stops at VRXR.start like it always did.
function VR.supported()
  -- visionOS reports its OS as "iOS" on purpose (the engine port keeps every
  -- existing iOS branch working), so the backend's own answer is the reliable
  -- one and the OS string is only a fallback for the headless test suite.
  if VRBackend.available() then return true end
  local ok, os = pcall(function() return love.system.getOS() end)
  if not ok or not os then return true end
  return os == "Windows"
end

function VR.backendKind()
  return VRBackend.kind()
end

function VR.enabled()
  return VR.supported() and VR.setting:get() == true
end

function VR.active()
  return started and VRXR.isRunning()
end

function VR.status()
  if not VR.enabled() then return "off" end
  if failed then return failed end
  return VRXR.status()
end

-- Let go of every input this module was holding: the GB buttons pressed
-- through the overlay path, and the synthetic left stick. Runs when the
-- session ends and whenever a frame has no controller state to read.
local function releaseInputs()
  local ok, Game = pcall(require, "src.core.Game")
  if not ok or not Game.input then return end
  for btn in pairs(held) do
    pcall(function() Game.input:overlayReleased(btn) end)
    held[btn] = nil
  end
  pcall(function()
    Game.input:gamepadaxis(nil, "leftx", 0)
    Game.input:gamepadaxis(nil, "lefty", 0)
  end)
  lastHandY = nil
end

local function shutdown(reason)
  if started then
    VRXR.stop()
    started = false
  end
  if savedVsync ~= nil then
    pcall(love.window.setVSync, savedVsync)
    savedVsync = nil
  end
  -- the placed camera may still be a VR eye's; the orbit must get the
  -- pass back clean
  Voxel3D.camera = nil
  mirrorSrc = nil
  releaseInputs()
  BattleCam.still = false
  VoxelScene.spriteLean = nil
  Pokedex.clear()
  -- the horde's gun too: its VR frame is a matrix built from a hand pose,
  -- and a stale one left behind would pin the model to wherever the
  -- controller was when the session died -- on the FLAT screen, where the
  -- view model should have taken over
  V.require("HordeGun").clear()
  zoom, heightOff = 1, 0
  fpYawOff, snapArmed = 0, true
  camMode, fadeAlpha = "explore", 0
  status = reason or "off"
end

VR.shutdown = shutdown          -- named for the probe driver

-- Whether the flat screen is showing something the world pass cannot: a
-- menu, a dialog, a battle, a transition wipe. The quad and the pokedex's
-- screen both key on it.
local function uiShowing()
  local ok, showing = pcall(function()
    local Game = require("src.core.Game")
    local top = Game.stack and Game.stack:top()
    return top ~= Game.overworld
           or (Game.overworld and Game.overworld.transitioning) or false
  end)
  return ok and showing or false
end

-- ------- the pokedex's screen
--
-- What the device in the hand shows during a battle: the flat window --
-- which IS the 2D battle screen for as long as the battle state draws --
-- copied into a canvas the scene pass can texture with, cropped by UV to
-- the battle's own letterbox so the screen wears the GB frame edge to
-- edge. Menus over the battle (the party, the bag) ride along for free:
-- they are the flat screen too, and reading them on the device in your
-- hand is exactly the point.
local dexCanvas = nil

local function dexScreen()
  local ok, out = pcall(function()
    local ww, wh = love.graphics.getPixelDimensions()
    if not (ww and ww > 0 and wh and wh > 0) then return nil end
    if not (dexCanvas and dexCanvas:getWidth() == ww
            and dexCanvas:getHeight() == wh) then
      dexCanvas = love.graphics.newCanvas(ww, wh)
      pcall(dexCanvas.setFilter, dexCanvas, "nearest", "nearest")
    end
    local fbo = fboCache[dexCanvas]
    if not fbo then
      fbo = VRGL.canvasFBO(dexCanvas)
      fboCache[dexCanvas] = fbo
    end
    if not (fbo and VRGL.copyFrontToCanvas(fbo, ww, wh)) then return nil end
    local BattleScene = V.require("BattleScene")
    local lx, ly, s = BattleScene.letterbox()
    return { dexCanvas,
             lx / ww, ly / wh,
             (lx + BattleScene.GB_W * s) / ww,
             (ly + BattleScene.GB_H * s) / wh }
  end)
  return ok and out or nil
end

-- Declared here because renderWorld below is compiled before the panel
-- section defines it, and a local that is not yet in scope resolves to a nil
-- global instead -- which took the whole voxel pipeline down with it.
local updatePanel

-- A MIPMAPPED copy of the UI layer, for a screen seen at an angle.
--
-- The device's screen is tilted 45 degrees away, so one axis of the texture is
-- compressed hard against the pixel grid. Sampled with nearest and no mipmaps
-- -- which is right for a flat 160x144 frame filling a monitor -- every edge
-- lands on a different texel from one frame to the next and the whole readout
-- shimmers.
--
-- Mipmaps plus anisotropy is what that asks for: minification takes a filtered
-- level chosen per pixel, while MAGNIFICATION stays nearest so the art is
-- still crisp squares when the device is close. Anisotropy is the part that
-- matters at a slant -- an isotropic mip pick blurs the un-compressed axis
-- just as much, which is legible but soft.
--
-- The copy is 160x144. Redrawing it once a frame is nothing, and it leaves the
-- engine's own canvas untouched -- it is the flat screen's, and giving it
-- mipmaps would change what every other reader of it gets.
local uiMip = nil
local function uiMipped(src)
  if not src then return nil end
  local w, h = src:getWidth(), src:getHeight()
  if not (uiMip and uiMip:getWidth() == w and uiMip:getHeight() == h) then
    if uiMip then pcall(uiMip.release, uiMip) end
    local ok, c = pcall(love.graphics.newCanvas, w, h,
                        { mipmaps = "manual", dpiscale = 1 })
    if not ok then return src end
    uiMip = c
    -- LINEAR both ways, and the magnification is the one that matters here.
    --
    -- The screen is about 160 texels across and covers several hundred pixels
    -- of view at reading distance, so it is MAGNIFIED, not minified. Nearest
    -- magnification snaps each pixel to whichever texel its centre lands in --
    -- and on a surface tilted 45 degrees away, the smallest head movement
    -- moves that centre across a texel boundary. Every edge in the readout
    -- then flips from one frame to the next. That is the flicker, and it is
    -- why filtering the minification side changed nothing.
    --
    -- The cost is softer pixel edges up close. On a slanted screen the art is
    -- never on the pixel grid anyway, so there was no crispness there to keep.
    pcall(uiMip.setFilter, uiMip, "linear", "linear", 16)
    pcall(uiMip.setMipmapFilter, uiMip, "linear")
    pcall(uiMip.setWrap, uiMip, "clamp", "clamp")
  end
  local ok = pcall(function()
    love.graphics.push("all")
    love.graphics.setCanvas(uiMip)
    love.graphics.origin()
    love.graphics.setScissor()
    love.graphics.setShader()
    love.graphics.setDepthMode()
    -- OPAQUE, and that is the actual fix for the shimmer.
    --
    -- The scene shader discards any texel with alpha below 0.5 -- right for
    -- sprite sheets, where it keeps a transparent border out of the depth
    -- buffer. The UI layer is transparent everywhere it drew nothing, so every
    -- glyph and every box edge is an alpha edge, and on a screen tilted away
    -- from the eye each of those flips between drawn and discarded as the
    -- sampling point moves. That is the flicker; it is an alpha test, not a
    -- colour filter, which is why mipmaps alone did nothing for it.
    --
    -- A screen is opaque anyway. Cleared to the Game Boy's own off-white, the
    -- whole quad passes the test, nothing flips, and the mipmaps and
    -- anisotropy below finally have plain colour to work on.
    love.graphics.clear(0.94, 0.98, 0.91, 1)
    love.graphics.setBlendMode("alpha")
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(src, 0, 0)
    love.graphics.setCanvas()
    love.graphics.pop()
  end)
  if not ok then return src end
  pcall(uiMip.generateMipmaps, uiMip)
  return uiMip
end

-- Where the device sits when nobody is holding it: below the line of sight,
-- turned back up toward the face.
--
-- 45 degrees down is the angle a thing you are reading actually sits at, and
-- it keeps the device out of the view while walking -- a screen pinned in
-- front of the eyes is the thing every headset UI gets wrong. HELD_DIST is
-- reading distance, not arm's length: the device is small and its screen is
-- 160x144.
local HELD_PITCH = -math.pi / 4
local HELD_DIST = 0.5

-- Builds a hand-like pose from the head's, for Pokedex.place.
--
-- Only the HEADING carries over. A device that pitched and rolled with the
-- head would swing under every glance and be unreadable -- and it is the head
-- moving relative to it that has to do the reading, not the other way round.
local function heldPose(head)
  local q = head.quat
  local R = Mat4.fromQuat(q[1], q[2], q[3], q[4])
  -- R's third column is +Z; forward is its negation, flattened to the heading.
  local fx, fz = -R[3], -R[11]
  local flen = math.sqrt(fx * fx + fz * fz)
  if flen < 1e-6 then fx, fz, flen = 0, -1, 1 end
  fx, fz = fx / flen, fz / flen

  -- Down the view by HELD_PITCH: forward shortens by the cosine, and the
  -- device drops by the sine.
  local cp, sp = math.cos(HELD_PITCH), math.sin(HELD_PITCH)
  local pos = {
    head.pos[1] + fx * HELD_DIST * cp,
    head.pos[2] + HELD_DIST * sp,
    head.pos[3] + fz * HELD_DIST * cp,
  }

  -- Facing the same heading, pitched back up so the screen looks at the face.
  -- Ry(a) sends (0,0,-1) to (-sin a, 0, -cos a), which solves the heading;
  -- the pitch is then a rotation about the turned X axis, composed as
  -- quaternions so Pokedex.place's own TILT still applies on top.
  local a = math.atan2(-fx, -fz)
  local hy, hw = math.sin(a * 0.5), math.cos(a * 0.5)
  local px, pw = math.sin(-HELD_PITCH * 0.5), math.cos(-HELD_PITCH * 0.5)
  return {
    pos = pos,
    quat = { hw * px, hy * pw, -hy * px, hw * pw },
  }
end

-- ------- the world, once per eye

-- Whether the world could be drawn RIGHT NOW.
--
-- Split out of renderWorld because the answer has to be known BEFORE the
-- session is claimed, not after. On CompositorServices, claiming stands the
-- host's own renderer down -- so a mod that takes the frame loop before it can
-- draw anything submits empty frames, and the player sits in a black space
-- until the chunk queue catches up. That window is not small: the first map
-- has to mesh, and this port deliberately runs a tighter meshing budget in a
-- headset (lib/ChunkMesher.lua) precisely so it does not miss frames.
--
-- On OpenXR it never showed, because nothing stands down there -- the flat
-- window keeps drawing and the headset simply shows black for a moment.
local function worldRenderable()
  local ok, Game = pcall(require, "src.core.Game")
  local ow = ok and Game.overworld or nil
  if not (ow and ow.map and ow.camera and Voxel.active()
          and Voxel3D.available()) then
    return false
  end
  -- The meshes themselves, not just the modules: VoxelScene.render returns
  -- nil until the current map has terrain, and a nil frame is exactly the
  -- empty submission this is here to prevent.
  local okP, terrain = pcall(function()
    return (VoxelScene.prefetch(ow))
  end)
  return okP and terrain ~= nil and terrain ~= false
end

-- Whether this renderer's canvases store the frame upside down.
--
-- lib/Voxel3D.lua premultiplies a clip-space Y flip onto every projection,
-- which is right for an OpenGL canvas and wrong for a Metal one -- the frame
-- comes out mirrored and is turned back over at the very end. Anything that
-- reasons in CANVAS ROWS is therefore inverted on Metal, and the sky's ray fan
-- is exactly that: its v runs top-to-bottom across the canvas.
--
-- Kept here rather than in VRRig, which is deliberately platform-free and
-- stays that way.
local metalCanvases = nil
local function canvasIsFlipped()
  if metalCanvases == nil then
    local ok, name = pcall(love.graphics.getRendererInfo)
    metalCanvases = ok and name == "Metal"
  end
  return metalCanvases
end

-- Turn a sky ray fan over in v: the far end becomes the base and the step
-- reverses. base + 1*dv is the v = 1 edge, so that is the new v = 0.
local function flipSkyRay(ray)
  if not (ray and ray.base and ray.dv) then return ray end
  return {
    base = { ray.base[1] + ray.dv[1],
             ray.base[2] + ray.dv[2],
             ray.base[3] + ray.dv[3] },
    du = ray.du,
    dv = { -ray.dv[1], -ray.dv[2], -ray.dv[3] },
  }
end

local function renderWorld(views, ctl)
  local ok, Game = pcall(require, "src.core.Game")
  local ow = ok and Game.overworld or nil
  if not (ow and ow.map and ow.camera and Voxel.active()
          and Voxel3D.available()) then
    return false
  end
  local vw, vh = 320, 288
  pcall(function() vw, vh = Game.renderer:worldViewSize() end)

  -- Whatever the camera does, the CARDS hold the top rung's near-upright
  -- lean: a head that roams has no one pitch for them to match, and 75
  -- degrees is the pose that reads as "standing" from anywhere. Cleared
  -- on shutdown, so the flat screen leans with the rung as ever.
  VoxelScene.spriteLean = math.rad(75)

  local pivot, anchor, scale, mountYaw
  local fp = FirstPerson.engaged()
  local battle, battleFloor
  if camMode == "battle" then battle, battleFloor = battleStage() end
  if battle then
    -- the over-the-shoulder seat the flat battle shot stands in, pulled
    -- close enough for a headset's own lens (see VRRig.battleMount), at
    -- life scale, turned to face the arena
    local rec = BattleCam.rig(battle, battleFloor)
    pivot, mountYaw = VRRig.battleMount(rec.eye, rec.focus)
    anchor = { 0, 0, 0 }
    scale = VRRig.FP_SCALE
  elseif fp then
    local p = ow.player
    local gh = 0
    pcall(function() gh = VoxelScene.groundAt(ow.map, p.cellX, p.cellY) end)
    pivot = VRRig.fpPivot(p.px, p.py, gh, FirstPerson.EYE_HEIGHT)
    anchor = { 0, 0, 0 }
    scale = VRRig.FP_SCALE
    -- the snap turn is a yaw on the MAPPING, same seam the battle mount
    -- turns through
    if fpYawOff ~= 0 then mountYaw = fpYawOff end
    -- the HMD is the head: its yaw and pitch (plus the snaps) become
    -- FirstPerson's, so FreeMove walks where you look and A talks to
    -- what you face
    local yaw, pitch = VRRig.headYawPitch(views[1].pose.quat)
    FirstPerson.yaw = wrapPi(yaw + fpYawOff)
    FirstPerson.pitch = math.max(FirstPerson.PITCH_UP,
                          math.min(FirstPerson.PITCH_DOWN, pitch))
  else
    -- The table presents the world exactly as the flat screen does at
    -- rest: the pivot sits VIEW_DIST away along the RUNG'S own angle
    -- (stepping rungs re-tilts the model, easing with the rung tween),
    -- at the scale that reproduces the flat framing -- then the player's
    -- own adjustments go on top: the stick's zoom, the grip's height.
    pivot = VRRig.dioramaPivot(ow.camera.x + vw / 2, ow.camera.y + vh / 2)
    anchor = VRRig.dioramaAnchor(Voxel.angle, heightOff)
    scale = VRRig.dioramaScale(vh, Voxel.FOCAL) / zoom
  end

  -- The pokedex, on the tracked left hand, under this very mapping --
  -- but only where it earns its keep: FIRST PERSON, where its screen is
  -- every menu, dialog and wipe the flat screen shows (and the floating
  -- billboard is retired outright -- see updateQuad), and the BATTLE
  -- seat, where its screen is the fight's own 2D scene. The diorama
  -- does without: a hand-sized device hovering over a tabletop town is
  -- clutter, and the panel serves there. No hand tracked, no device.
  --
  -- ...unless there is no hand to track. A DualSense reports no pose at all,
  -- and on that pad the device would simply never appear -- so one is HELD for
  -- the player, at a fixed place below the line of sight. Reading it is then
  -- the same gesture as reading a real one: glance down.
  local hand = ctl and ctl.handl or nil
  -- A MENU outside a fight and outside first person -- the title screen, the
  -- options, anything the diorama is behind -- goes on the device too, and on
  -- the HELD one even when a hand is being tracked.
  --
  -- Held rather than hand-mounted on purpose: a menu is read, not glanced at.
  -- Following the wrist means the text moves while it is being read, and the
  -- hand has to be held up for as long as the menu is open. The fixed place
  -- below the line of sight is the pad's arrangement, and for a menu it is the
  -- better one whether or not there is a hand to mount it on.
  local menu = uiShowing() and not (battle or fp)
  if (menu or not hand) and VRXR.hasQuadLayer == false then
    hand = heldPose(views[1].pose)
  end
  if hand and (battle or fp or menu) then
    Pokedex.place(hand, pivot, anchor, scale, mountYaw)
    if uiShowing() then
      -- The engine's UI layer where there is no front buffer to read.
      --
      -- dexScreen copies the window's front buffer with raw GL, which exists
      -- on Windows and nowhere else. The renderer's own canvas is the same
      -- content and better: the 2D pass alone, at its native 160x144, already
      -- transparent where it drew nothing -- so the device shows a menu rather
      -- than a menu inside a picture of the world.
      local scr = nil
      if VRXR.hasQuadLayer == false then
        local okG, Game = pcall(require, "src.core.Game")
        local c = okG and Game.renderer and Game.renderer.canvas or nil
        c = uiMipped(c)
        if c then scr = { c, 0, 0, 1, 1 } end
      else
        scr = dexScreen()
      end
      if scr then
        Pokedex.screen(scr[1], scr[2], scr[3], scr[4], scr[5])
      end
    elseif V.require("Horde").active then
      -- HORDE MODE's readout, on the device already in the player's left
      -- hand. It cannot be a flat overlay: the eye buffers have
      -- ASYMMETRIC frusta, so the same canvas pixel is a different ANGLE
      -- in each eye and a 2D HUD drawn into both tears down the middle.
      -- The Pokedex is real geometry both eyes see from their own
      -- position, so the stereo is correct by construction -- and it is
      -- already tracked, already lit, and already the thing this mod
      -- puts information on. (The gun wore it briefly and that was
      -- worse: a screen on the slide sits exactly where the iron sights
      -- need to be looked through.)
      --
      -- The UV rect goes over the usual way up: v = 0 at the TOP, which
      -- is how the device's screen quad reads every other texture it
      -- wears. An inverted rect was tried first, on the theory that a
      -- self-drawn canvas samples from the bottom -- it does not here,
      -- and it stood the readout on its head.
      local tex = V.require("HordeHud").panelTexture()
      if tex then Pokedex.screen(tex, 0, 0, 1, 1) end
    end
  else
    Pokedex.clear()
  end

  -- and the horde's gun on the tracked RIGHT hand, under the same
  -- mapping. The AIM pose where the runtime offers one -- the barrel
  -- should point where the player is pointing, not along their wrist --
  -- and the grip pose as the fallback. Placed here rather than in the
  -- draw because the shot is traced down the model's own axis, so the
  -- matrix has to exist before anything can be hit with it.
  do
    local HordeGun = V.require("HordeGun")
    local right = ctl and (ctl.aimr or ctl.handr) or nil
    if right and fp and not battle and V.require("Horde").active then
      HordeGun.place(right, pivot, anchor, scale, mountYaw)
    else
      HordeGun.clear()
    end
  end

  local eyes = {}
  for i = 1, 2 do
    local v = views[i]
    local target = VRXR.eyeCanvas and VRXR.eyeCanvas(i) or nil
    local targetDepth = VRXR.eyeDepth and VRXR.eyeDepth(i) or nil
    local rateMap = VRXR.rateMap and VRXR.rateMap(i) or nil
    local pw, ph = nil, nil
    if VRXR.eyePhysicalSize then pw, ph = VRXR.eyePhysicalSize(i) end
    eyes[i] = {
      camera = (function()
        local cam = VRRig.eyeCamera(v.pose, v.fov, pivot, anchor, scale, mountYaw)
        -- The sun stays where it is when the head nods, instead of riding
        -- along with it, once the fan agrees with the canvas it is read in.
        if cam and cam.skyRay and canvasIsFlipped() then
          cam.skyRay = flipSkyRay(cam.skyRay)
        end
        return cam
      end)(),
      -- visionOS lends a foveated intermediary whose reported dimensions are
      -- logical screen pixels; its physical allocation is much smaller.
      w = target and target:getWidth() or v.w,
      h = target and target:getHeight() or v.h,
      slot = i == 1 and "vrL" or "vrR",
      -- OpenXR has no Canvas targets; CompositorServices supplies matching
      -- foveated colour and readable depth intermediaries.
      target = target,
      depth = targetDepth,
      rateMap = rateMap,
      physW = pw, physH = ph,
      -- Called by the water pass, which reads the frame in screen space and
      -- therefore needs the de-foveated snapshot rather than the packed
      -- attachments.
      resolve = VRXR.resolveEye and function() return VRXR.resolveEye(i) end
                or nil,
      -- the battle seat is a placed shot, not the first-person rig: the
      -- cards keep their stage lean rather than yawing at this eye, and
      -- the player's own card stays visible in it
      adopt = not battle,
    }
  end
  eyes.cx, eyes.cy = pivot[1], pivot[3]


  -- Forced to identity for the eye pass regardless of what the flat frame
  -- left behind. This pass measures in canvas pixels; anything the game had
  -- pushed is meaningless here and can only shrink or shift it.
  pcall(love.graphics.origin)
  pcall(love.graphics.setScissor)

  -- The world extent an EYE sees, not the one a Game Boy screen sees.
  --
  -- vw, vh are WORLD PIXELS: they say how much world this view covers, and
  -- VoxelScene uses them to decide which chunks to ask ChunkMesher for. The
  -- flat path passes 180x320, the size of the handheld's screen. An eye looks
  -- at the same world through a 95-degree frustum and sees several times that
  -- -- so everything outside the little 180x320 box was never REQUESTED, and
  -- came out black with chunk-shaped edges. Turning your head while a map
  -- loaded turned the black with it, because the hole is world-locked: that
  -- is what finally identified it, after six measurements of the texture, the
  -- transform and the blit all came back clean.
  --
  -- Derived, not guessed: keep the flat path's density of world pixels per
  -- screen pixel (1080/180 = 6) and apply it to the eye texture. Voxel3D.cell
  -- is exactly that ratio, so the sky's dither grid and the water's stay the
  -- size they are on a flat screen instead of changing with the headset.
  local rw, rh = vw, vh
  if eyes[1].target and views[1].w and views[1].h then
    local density = 6
    pcall(function()
      local sw = select(1, Game.renderer:worldViewSize())
      local pw = love.graphics.getWidth()
      if sw and sw > 0 and pw and pw > 0 then density = pw / sw end
    end)
    if density <= 0 then density = 6 end
    rw = math.max(vw, math.floor(views[1].w / density))
    rh = math.max(vh, math.floor(views[1].h / density))
  end


  -- The panel, before the eyes draw it. Only where there is no composition
  -- layer to submit it on; the layer path places it after the frame instead.
  if VRXR.hasQuadLayer == false then
    updatePanel(true, fp, views[1].pose, pivot, anchor, scale, mountYaw)
  end

  local okR, canvases = pcall(VoxelScene.render, ow, 0, 0, rw, rh,
                              VR.paletteFor, eyes)

  if not (okR and type(canvases) == "table" and canvases[1] and canvases[2])
  then
    return false
  end

  -- the snap's fade, over the finished eyes: plain black at this moment's
  -- strength, drawn before the blit so the headset never sees the swap.
  -- A full-frame fill is the ONE 2D thing that is safe to draw into an
  -- eye buffer -- it covers everything, so it does not matter that the
  -- two frusta disagree about where any given pixel points.
  if fadeAlpha > 0 then
    pcall(function()
      for i = 1, 2 do
        local c = canvases[i]
        love.graphics.setCanvas(c)
        love.graphics.setColor(0, 0, 0, math.min(1, fadeAlpha))
        love.graphics.rectangle("fill", 0, 0, c:getWidth(), c:getHeight())
      end
      love.graphics.setCanvas()
      love.graphics.setColor(1, 1, 1, 1)
    end)
  end

  for i = 1, 2 do
    local canvas = canvases[i]
    -- Nothing to copy when the scene already drew into the compositor's own
    -- texture; VRGL is nil on that path in any case.
    local tex, tw, th = VRGL and VRXR.acquireEye(i) or nil
    if tex then
      local fbo = fboCache[canvas]
      if not fbo then
        fbo = VRGL.canvasFBO(canvas)
        fboCache[canvas] = fbo
      end
      if fbo then
        VRGL.blitToTexture(fbo, canvas:getWidth(), canvas:getHeight(),
                           tex, tw, th)
      end
    end
    VRXR.releaseEye(i)
  end
  mirrorSrc = canvases[1]
  return true
end

-- ------- the UI panel

-- Whether the flat screen is showing something the world pass cannot: a
-- menu, a dialog, a battle, a transition wipe -- or everything, when the
-- world pass is off entirely.
local function wantQuad(worldUp)
  if not worldUp then return true end
  return uiShowing()
end

-- The panel, where the backend has no composition layer to submit it on.
--
-- Same question as updateQuad answers -- is the flat screen saying anything --
-- and the same letterbox rect. What differs is only the destination: there is
-- no quad to hand the runtime here, so the picture becomes a surface standing
-- in the scene (lib/VRPanel.lua). Returns nil always, because nothing is
-- submitted; the caller's endFrame has no layer to place.
function updatePanel(worldUp, fp, pose, pivot, anchor, scale, yaw)
  local VRPanel = V.require("VRPanel")
  if not wantQuad(worldUp) then VRPanel.clear() return nil end
  -- Same exemption as the quad path: where the pokedex is up and lit it IS
  -- the screen, and a second copy floating beside it is clutter.
  if Pokedex.frame and Pokedex.frame.tex then VRPanel.clear() return nil end

  -- The engine's UI LAYER, not the finished frame.
  --
  -- This is what the OpenXR build effectively shows: the 2D pass on its own --
  -- menus, dialogs, the battle screen -- at its native 160x144 and transparent
  -- everywhere it drew nothing. The world is not in it.
  --
  -- The whole flat frame was the obvious thing to reach for and was wrong
  -- twice over. It carries the world as well, so the panel showed a picture of
  -- the scene it was standing in; and because that frame is where the panel is
  -- composited on the flat path, the two fed each other into a tunnel of
  -- receding copies. A layer that never contains the panel cannot do that.
  --
  -- No crop either: the canvas IS the frame, so the letterbox arithmetic the
  -- quad path needs has nothing to cut here.
  local tex = nil
  pcall(function()
    local okG, Game = pcall(require, "src.core.Game")
    if okG and Game.renderer then tex = Game.renderer.canvas end
  end)
  -- The virtual screen is the fallback for an engine without that canvas.
  if not tex and VRXR.screenTexture then tex = VRXR.screenTexture() end
  if not tex then VRPanel.clear() return nil end

  local u0, v0, u1, v1 = 0, 0, 1, 1

  VRPanel.place(tex, u0, v0, u1, v1, pose, pivot, anchor, scale, yaw, fp)
  return nil
end

local function updateQuad(worldUp, fp)
  if not wantQuad(worldUp) then return nil end
  -- Wherever the pokedex is up and lit -- first person's menus, the
  -- battle seat's 2D fight -- it IS the screen, and no floating
  -- billboard is submitted at all. (No tracked left hand still gets
  -- the panel: the UI must be readable somewhere.)
  if Pokedex.frame and Pokedex.frame.tex then return nil end
  local tex, qw, qh = VRXR.acquireQuad()
  if not tex then return nil end
  local ww, wh = qw, qh
  pcall(function() ww, wh = love.graphics.getPixelDimensions() end)
  -- The panel wears the GB FRAME, not the window: everything the flat
  -- screen has to say lives in the 160x144 letterbox (the world around
  -- it is just the mirror's picture). The frame region is blitted OUT
  -- of the window and SCALED into the swapchain image -- never copied
  -- pixel-for-pixel, because the swapchain's size is fixed at session
  -- start and a fullscreened window outgrows it, running the frame (and
  -- the START menu flush with its right edge) off the copy. Scaled, the
  -- panel shows the identical picture at the identical ratio whatever
  -- size the window is. Source coordinates are GL's, origin bottom-left.
  local crop = nil
  local copied = false
  pcall(function()
    local BattleScene = V.require("BattleScene")
    local lx, ly, s = BattleScene.letterbox()
    local wpx = math.ceil(BattleScene.GB_W * s)
    local hpx = math.ceil(BattleScene.GB_H * s)
    local sx = math.max(0, math.floor(lx))
    local sy = math.max(0, math.floor(wh - ly - hpx))
    wpx = math.min(wpx, ww - sx)
    hpx = math.min(hpx, wh - sy)
    if wpx < 1 or hpx < 1 then return end
    -- fitted to the swapchain image at the REGION's own aspect: the
    -- crop then presents exactly that rect, so the panel's shape is the
    -- GB frame's at any window and any swapchain size
    local fit = math.min(qw / wpx, qh / hpx)
    local dw = math.max(1, math.floor(wpx * fit))
    local dh = math.max(1, math.floor(hpx * fit))
    if VRGL.copyFrontRegionToTexture(tex, sx, sy, wpx, hpx, dw, dh) then
      copied = true
      crop = { 0, 0, dw, dh }
    end
  end)
  if not copied then
    -- no letterbox to cut (or the blit refused): the old whole-window
    -- copy, clamped, is still a readable panel
    VRGL.copyFrontBuffer(tex, math.min(qw, ww), math.min(qh, wh))
  end
  VRXR.releaseQuad()
  local base = fp and QUAD_FP or QUAD_DIORAMA
  if not crop then return base end
  return { pos = base.pos, width = base.width, crop = crop }
end

-- ------- the controllers
--
-- The mapping the mod ships (rebindable in the runtime's own UI):
--
--   both modes    left stick moves (through the engine's own stick path,
--                 so it grid-walks the diorama and free-walks 1ST);
--                 A/B are A/B; either trigger is START; clicking the
--                 LEFT stick steps the VOXEL angle ladder exactly as
--                 the "3" key (and the pad's SELECT) does.
--   1ST only      right stick left/right SNAP-TURNS 45 degrees a flick.
--   diorama only  right stick up/down zooms the model; squeezing a grip
--                 and moving that hand up or down drags the whole table
--                 with it.
--
-- Leaving VR is the VR row's job alone (OPTIONS menu or the manager) --
-- no controller button does it. VR.leave below stays as the API for it.

-- The left stick click makes EXACTLY the step the "3" key makes: one
-- rung up the VOXEL angle ladder, wrapping, stepping over FULL, clearing
-- TILT and GBC FX in the save -- by calling the very function the key
-- and the pad's SELECT button already share. main.lua installs it below
-- (cycleVoxel is a local of that file); the free-roam gate is the
-- registry's own, inside it, so a click over a menu or mid-warp is a
-- no-op exactly like the key.
VR.cycleVoxel = nil             -- cycleVoxel(game), set by main.lua

function VR.stepView()
  pcall(function()
    if not VR.cycleVoxel then return end
    VR.cycleVoxel(require("src.core.Game"))
  end)
end

-- Leave VR: the VR row toggled back off and persisted, exactly as if
-- stepped on the OPTIONS menu, so the next update tears the session down
-- and the flat screen takes the picture back. Deliberately bound to NO
-- controller button (a click that ejects you from the headset is a trap
-- mid-fight); kept as the one programmatic door out.
function VR.leave()
  pcall(function()
    local Game = require("src.core.Game")
    VR.setting:setIndex(VR.setting:read() + 1, Game)
  end)
end

local function setGB(inp, btn, down)
  if down and not held[btn] then
    held[btn] = true
    inp:overlayPressed(btn)
  elseif not down and held[btn] then
    held[btn] = nil
    inp:overlayReleased(btn)
  end
end

local function driveControls(ctl, dt, fp)
  if not ctl then
    releaseInputs()
    return
  end
  local ok, Game = pcall(require, "src.core.Game")
  if not (ok and Game.input) then return end
  local inp = Game.input

  -- HORDE MODE re-reads the right hand as a weapon: the trigger fires
  -- (its own OpenXR action, suggested alongside START on the same input
  -- -- see VRXR.setupInput), and B reloads. START is dropped rather than
  -- forwarded, because the mode does not pause. Everything else -- the
  -- stick's walk, the snap turn, A -- keeps working, so the player can
  -- still move and look while they are being chased.
  local Horde = V.require("Horde")
  if Horde.playing() then
    local Gun = V.require("HordeGun")
    if ctl.fireChanged and ctl.fire then Gun.fire() end
    if ctl.bChanged and ctl.b then Gun.reload() end
    setGB(inp, "a", ctl.a)
    setGB(inp, "b", false)
    setGB(inp, "start", false)
  else
    setGB(inp, "a", ctl.a)
    setGB(inp, "b", ctl.b)
    setGB(inp, "start", ctl.start)
  end

  -- the left stick, through the engine's OWN stick handler: it quantises
  -- to the grid d-pad for the diorama, and FirstPerson.moveVector reads
  -- the same raw pair for the free walk. OpenXR's +Y is up; the engine's
  -- lefty is +down.
  inp:gamepadaxis(nil, "leftx", ctl.moveX or 0)
  inp:gamepadaxis(nil, "lefty", -(ctl.moveY or 0))

  -- the left stick click: the VOXEL ladder ordinarily, and the way out of
  -- horde mode while it runs (the rung is locked there, so the click has
  -- nothing else to do, and a headset has no ESCAPE key)
  if ctl.toggleChanged and ctl.toggle then
    if Horde.active then Horde.askExit() else VR.stepView() end
  end

  -- first person's turn on the right stick. SMOOTH TURN ON makes it a
  -- rate -- hold and the world rotates under you -- and OFF (the
  -- default) makes it a 45-degree snap per flick: see the row's own
  -- reasoning where it is declared. Either way the offset turns the
  -- MAPPING, so the eyes, the walk direction, the pokedex and the gun
  -- all agree about which way the world now faces.
  if fp and camMode ~= "battle" and VR.smoothTurn:get() == true then
    local sx = ctl.lookX or 0
    local a = math.abs(sx)
    if a > 0.2 then
      a = (a - 0.2) / 0.8
      -- increasing yaw turns LEFT in this mod's compass, so a stick
      -- pushed right subtracts -- the same sign the snap below uses
      fpYawOff = wrapPi(fpYawOff
                        - (sx > 0 and 1 or -1) * a * a
                          * VR.SMOOTH_TURN_RATE * (dt or 0))
    end
    snapArmed = true       -- so a switch back to snap mid-flick re-arms
  elseif fp and camMode ~= "battle" then
    local sx = ctl.lookX or 0
    if math.abs(sx) > 0.65 then
      if snapArmed then
        snapArmed = false
        -- increasing yaw turns LEFT in this mod's compass, so a stick
        -- pushed right subtracts
        fpYawOff = wrapPi(fpYawOff + (sx > 0 and -SNAP_TURN or SNAP_TURN))
      end
    elseif math.abs(sx) < 0.35 then
      snapArmed = true
    end
  end

  if not fp and camMode ~= "battle" then
    local zy = ctl.lookY or 0
    if math.abs(zy) > 0.15 then
      zoom = math.max(0.35, math.min(4, zoom * math.exp(zy * (dt or 0) * 1.6)))
    end
    -- the grab-drag: while a grip is squeezed, the table follows that
    -- hand's height, metre for metre
    local gl, gr = ctl.gripL or 0, ctl.gripR or 0
    local y = (gr >= gl) and ctl.handrY or ctl.handlY
    if math.max(gl, gr) > 0.6 and y then
      if lastHandY then
        heightOff = math.max(-1.5, math.min(1.5, heightOff + (y - lastHandY)))
      end
      lastHandY = y
    else
      lastHandY = nil
    end
  else
    lastHandY = nil
  end
end

-- ------- the hands, with VR switched off
--
-- The same gestures, driving the flat screen: the launcher window shows the
-- game before anyone has entered the space, and inside the space the VR row
-- can still be off. Both are states a player sits in, and in both of them
-- raising a hand did nothing at all.
--
-- Narrower than driveControls on purpose. What survives the flat screen is
-- exactly the four things it has a meaning for -- A, B, START and the walk --
-- and the rest of that function is about a rig that is not running here: the
-- snap turn, the zoom, the height grab and the VOXEL ladder all move state
-- the flat renderer never reads, so driving them from here would be writing
-- into the dark.
--
-- Silent whenever no hand is tracked, rather than releasing every frame: the
-- pad and the keyboard own these same buttons while the hands are down, and a
-- release per frame would take every press away from them again. The one
-- release happens on the way down, once.
local flatHeld = false

local function driveFlat()
  if not VRXR.handInput then return end
  local ok, ctl = pcall(VRXR.handInput)
  if not ok then return end

  if not ctl then
    if flatHeld then
      flatHeld = false
      releaseInputs()
    end
    return
  end
  flatHeld = true

  local okG, Game = pcall(require, "src.core.Game")
  if not (okG and Game.input) then return end
  local inp = Game.input
  setGB(inp, "a", ctl.a)
  setGB(inp, "b", ctl.b)
  setGB(inp, "start", ctl.start)
  -- OpenXR's +Y is up; the engine's lefty is +down. Same negation
  -- driveControls makes, for the same reason.
  inp:gamepadaxis(nil, "leftx", ctl.moveX or 0)
  inp:gamepadaxis(nil, "lefty", -(ctl.moveY or 0))
end

-- ------- the per-frame drive
--
-- Split in two so the hands cannot fall down a crack. updateRig returns true
-- only on the path that actually reached driveControls; every other way out --
-- and there are eleven of them -- is a frame where the rig drove nothing, and
-- the flat screen is what the player is looking at.
--
-- That distinction is the whole bug this split fixes. VR read ON in the
-- settings from the last session, so update() went down the VR branch and sat
-- in "the world is still building" with no immersive space to claim, while the
-- flat drive lived on the OFF branch and was never reached. Switched on, no
-- session, flat picture, dead hands.

local updateRig

function VR.update(dt)
  if updateRig(dt) then
    -- the rig drove this frame, so the flat path must not press the same
    -- button behind it -- and must not release it either
    flatHeld = false
    return
  end
  driveFlat()
end

function updateRig(dt)
  local on = VR.enabled()
  if not on then
    if wasOn then
      shutdown("off")
      failed = nil
    end
    wasOn = false
    return false
  end

  if not wasOn then failed = nil end   -- a fresh toggle earns a fresh try
  wasOn = true
  if failed then return end

  -- Nothing is claimed until there is something to show. See worldRenderable.
  if not started and VRXR.resumable and not worldRenderable() then
    if waiting ~= "world" then
      waiting = "world"
      print("[DRAMATIC_SHAPE] VR waiting: the world is still building")
    end
    return
  end

  if not started then
    local qw, qh = 1024, 768
    pcall(function() qw, qh = love.graphics.getPixelDimensions() end)
    if VRXR.start(qw, qh) then
      started = true
      waiting = nil
      status = "session created"
      print("[DRAMATIC_SHAPE] VR: " .. VRXR.status())
    elseif VRXR.resumable then
      -- A backend whose session can come back on its own: a failed start is
      -- "not yet", not "not ever", so it must not latch.
      --
      -- On visionOS the compositor layer exists only while the immersive
      -- space is open, and the player opens and closes that with the Digital
      -- Crown as a matter of course -- each re-entry brings a NEW layer.
      -- Latching turned the Crown into a one-way door: VR read ON in the
      -- settings, the mod had given up, and only toggling the row off and on
      -- brought it back.
      local why = VRXR.status()
      if why ~= waiting then
        waiting = why
        print("[DRAMATIC_SHAPE] VR waiting: " .. why)
      end
      return
    else
      failed = VRXR.status()
      print("[DRAMATIC_SHAPE] VR unavailable: " .. failed
            .. " -- fix that, then toggle the VR row to retry")
      return
    end
  end

  if not VRXR.poll() then
    -- the runtime took the session away (headset off, runtime shut down)
    shutdown("session lost")
    if VRXR.resumable then
      -- Not a failure here: the space was closed, and the next start() picks
      -- up the layer the next one brings.
      waiting = nil
      return
    end
    failed = "session lost -- toggle VR off and on to retry"
    return
  end
  if not VRXR.isRunning() then return end

  -- the headset paces the app now; vsync would fight it
  if savedVsync == nil then
    savedVsync = 1
    pcall(function() savedVsync = love.window.getVSync() end)
    pcall(love.window.setVSync, 0)
    -- And the frame cap with it. waitFrame is the clock in a headset, so a
    -- MAX FPS of 30 does not save power, it makes the compositor show every
    -- submitted frame three times -- each one warped a little further to the
    -- head's new pose. That reads as edges doubling while you turn, and it is
    -- invisible in a screen recording because the recording captures the
    -- frames, not the reprojections between them.
    -- The HIGHEST step, not 0: normalize() snaps to the nearest valid step,
    -- so 0 would come back as the lowest cap rather than no cap at all.
    pcall(function()
      local FrameCap = require("src.core.FrameCap")
      local steps = FrameCap.STEPS
      FrameCap.apply(steps[#steps])
    end)
  end

  -- the battle camera holds still for as long as a headset is watching:
  -- its drift is a flat screen's depth cue, and a swaying picture inside
  -- VR reads as the world lurching
  BattleCam.still = true

  -- The battle snap's fade: while the camera the frame WANTS is not the
  -- one it is showing, black rises; at full black the mount swaps; then
  -- black lifts. Driven here, on game time, so a fight that ends during
  -- the fade just turns it around.
  local want = battleStage() and "battle" or "explore"
  if want ~= camMode then
    fadeAlpha = math.min(1, fadeAlpha + (dt or 0) / FADE_TIME)
    if fadeAlpha >= 1 then camMode = want end
  else
    fadeAlpha = math.max(0, fadeAlpha - (dt or 0) / FADE_TIME)
  end

  local time, should = VRXR.waitFrame()


  if not time then return end

  -- the controllers, before the world renders: the frame the toggle
  -- flips rungs on should be the frame that renders the new rig. The
  -- state is kept in hand for renderWorld too -- the pokedex stands on
  -- the same frame's left-hand pose.
  local ctl = VRXR.input(time)
  driveControls(ctl, dt, FirstPerson.engaged())

  local worldUp = false
  if should then
    local views = VRXR.locateViews(time)
    if views then
      worldUp = renderWorld(views, ctl)
    end
  end
  -- Only the layer path runs here. The panel is placed inside renderWorld,
  -- before the eyes draw, because it IS drawn by them -- placing it after
  -- would show the previous frame's position, which on a surface that follows
  -- the head is a visible lag.
  local quadPose = nil
  if VRXR.hasQuadLayer ~= false then
    quadPose = updateQuad(worldUp, FirstPerson.engaged())
  end
  VRXR.endFrame(time, worldUp or nil, quadPose)
  return true
end

-- ------- the window while a headset owns the picture

-- The flat window becomes the mirror: the left eye, fitted to the window.
-- Returns nil when there is nothing to mirror (the caller draws the flat
-- path as ever).
function VR.mirror(sw, sh)
  if not (VR.active() and mirrorSrc) then return nil end
  if not (mirrorCanvas and mirrorCanvas:getWidth() == sw
          and mirrorCanvas:getHeight() == sh) then
    local ok, c = pcall(love.graphics.newCanvas, sw, sh)
    if not ok then return nil end
    mirrorCanvas = c
  end
  local ok = pcall(function()
    love.graphics.setCanvas(mirrorCanvas)
    love.graphics.clear(0, 0, 0, 1)
    local mw, mh = mirrorSrc:getDimensions()
    local s = math.min(sw / mw, sh / mh)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(mirrorSrc, (sw - mw * s) / 2, (sh - mh * s) / 2, 0, s, s)
    love.graphics.setCanvas()
  end)
  pcall(love.graphics.setCanvas)
  return ok and mirrorCanvas or nil
end

-- window resize, hot reload: the eye canvases are Voxel3D's and go with
-- its invalidate; ours is the mirror and the FBO ids learned from dead
-- canvases
function VR.invalidate()
  if mirrorCanvas and mirrorCanvas.release then
    pcall(mirrorCanvas.release, mirrorCanvas)
  end
  mirrorCanvas, mirrorSrc = nil, nil
  if dexCanvas and dexCanvas.release then pcall(dexCanvas.release, dexCanvas) end
  dexCanvas = nil
  Pokedex.invalidate()
  V.require("HordeGun").invalidate()
  V.require("HordeHud").invalidate()
  for k in pairs(fboCache) do fboCache[k] = nil end
end

return VR

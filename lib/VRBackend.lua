-- VRBackend -- picks the VR backend for the platform and hides the difference.
--
-- lib/VR.lua is written against VRXR's surface.  Rather than teach it about a
-- second one, this returns whichever backend applies and fills in the few
-- capabilities that genuinely differ, so VR.lua asks "does this backend have a
-- mirror?" instead of "am I on Windows?".
--
-- The two are not symmetric and pretending otherwise would be worse than the
-- honest asymmetry:
--
--   * OpenXR gives swapchain images that a canvas must be BLITTED into.
--     CompositorServices gives textures the scene renders into directly.
--   * OpenXR has composition layers, so the flat UI can be a quad the runtime
--     composites.  CompositorServices has none; that UI must become geometry.
--   * OpenXR has a mirror window.  On visionOS there is no window to mirror to.
--
-- Loading is lazy on purpose.  VRXR pulls in LuaJIT's FFI and tries to load
-- openxr_loader.dll; VRCS touches love.xr.  Requiring the wrong one on the
-- wrong platform is at best a wasted failure and at worst a crash inside a
-- render pipeline.

local V = ...

local VRBackend = {}

local backend = nil

local function detect()
  if backend then return backend end

  -- visionOS first: love.xr only exists there, so its presence is a stronger
  -- signal than any OS string. love.system.getOS() deliberately still reports
  -- "iOS" on visionOS (see mobile/visionos/README.md in the engine port), so
  -- asking it would give the wrong answer.
  if love.xr ~= nil and love.xr.available() then
    local ok, mod = pcall(V.require, "VRCS")
    if ok and mod then backend = mod; return backend end
  end

  local os = nil
  pcall(function() os = love.system.getOS() end)
  if os == "Windows" or os == nil then
    local ok, mod = pcall(V.require, "VRXR")
    if ok and mod then
      backend = mod
      backend.kind = backend.kind or "openxr"
      -- OpenXR's shape, stated here so VR.lua never has to special-case it.
      if backend.takesOverVSync == nil then backend.takesOverVSync = true end
      if backend.hasMirror == nil then backend.hasMirror = true end
      if backend.hasQuadLayer == nil then backend.hasQuadLayer = true end
      return backend
    end
  end

  return nil
end

VRBackend.get = detect

function VRBackend.kind()
  local b = detect()
  return b and b.kind or "none"
end

function VRBackend.available()
  return detect() ~= nil
end

return VRBackend

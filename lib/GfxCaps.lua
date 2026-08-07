-- The one question about the renderer this mod actually has to ask, asked
-- once.
--
-- THE FLIP. lib/Voxel3D.lua premultiplies Mat4.scale(1,-1,1) onto every
-- projection, because the mod bypasses LOVE's transform_projection and has to
-- reproduce the Y inversion LOVE applies to canvas projections itself. On
-- OpenGL that is exactly right. On Metal LOVE inverts nothing -- its render
-- targets are top-left already -- so on Metal the flip is one error, and every
-- place that turns the frame back over is the second error that cancels it.
--
-- That arrangement works and it has cost four bugs, every one of them found in
-- a headset rather than in a test: the water sampled the wrong rows, the sky
-- ray ran upside down, the sun and moon rode the wrong way, and the shadow map
-- was written by the rasteriser at one row and read by arithmetic at another.
-- Each was fixed where it surfaced, and each fix carried its own private copy
-- of `getRendererInfo() == "Metal"`. Four copies of one fact is three too
-- many: they can drift, and a fifth site simply forgets to ask.
--
-- So the fact lives here. Callers say what they mean -- rowsFlipped() -- and
-- the next Metal-versus-OpenGL discrepancy is one line in one file rather than
-- a hunt through five.
--
-- What this does NOT do is remove the flip. Deleting it on Metal and dropping
-- all the compensations with it is the real repair, and it is a wider change
-- than it looks: it moves the row arithmetic in Voxel3D's horizonY,
-- horizonLine, skyBody, project and drawWorldDisc, in the water shaders, and
-- in the host's own worldOverride blit -- which means it changes pixels in the
-- SHIPPED iOS build, on a path only a screenshot can check.

local GfxCaps = {}

local flipped = nil

-- Whether this renderer stores a canvas with its rows the other way up from
-- the OpenGL convention the mod's clip-space arithmetic is written against.
--
-- Cached: getRendererInfo is a driver query, this is asked per frame from
-- several passes, and the renderer does not change under a running process.
function GfxCaps.rowsFlipped()
  if flipped == nil then
    local ok, name = pcall(love.graphics.getRendererInfo)
    flipped = ok and name == "Metal"
  end
  return flipped
end

-- For the suite, which has to be able to exercise both answers.
function GfxCaps.forget()
  flipped = nil
end

return GfxCaps

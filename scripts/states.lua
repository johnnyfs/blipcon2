local pipeName = "states"
local pipe = io.open(pipeName, "w")

print("Starting states.lua")

local function on_emulation_exit()
    print("Exiting")
    if recording then
        pipe:close()
    end
end
emu.registerexit(on_emulation_exit)

function writeScreenToFile()
    img = gui.gdscreenshot()
    pipe:write(img)
end

function writeInputToFile()
    local inputTable = joypad.get(1)
    local inputString = string.format(
        "%d%d%d%d%d%d%d%d",
        inputTable["A"] and 1 or 0,
        inputTable["B"] and 1 or 0,
        inputTable["Select"] and 1 or 0,
        inputTable["Start"] and 1 or 0,
        inputTable["Up"] and 1 or 0,
        inputTable["Down"] and 1 or 0,
        inputTable["Left"] and 1 or 0,
        inputTable["Right"] and 1 or 0
    )
    pipe:write(inputString)
end

local ADVANCE_REWARD = 100.0 -- also game over penalty
local LIFE_REWARD = 10.0     -- for gain or loss of 1 life
local POWER_REWARD = 5.0     -- status up or down or starman
local COIN_REWARD = 1.0      -- for gain of 1 coin
local STEP_REWARD = 0.1      -- for advancing within a level
local POINT_REWARD = 0.001   -- 1 reward every 1000 points 
local NO_STEP_PENALTY = 0.01 -- for not advancing (per processed frame)

local prev_coins = 101
local prev_lives = -1
local prev_status = -1
local prev_rank = 256
local prev_points = 0
local prev_star_timer = 0
local prev_position = 0
local no_advance_timer = 0

local first = true
local last_frame = 0
local last_actions = nil

function getRewardSMB()
    -- Coins
    local coins = memory.readbyte(0x075E) * 1 + memory.readbyte(0x075F) * 10
    local coin_diff = 0
    if coins > prev_coins then
        coin_diff = coins - prev_coins
    end
    prev_coins = coins

    -- Lives
    local lives = memory.readbyte(0x075A)
    local life_diff = 0
    local game_over = 0
    if prev_lives == 1 and lives == 0 then
        game_over = 1
    elseif prev_lives ~= - 1 then
        life_diff = lives - prev_lives
    end
    prev_lives = lives

    -- Status
    local status = memory.readbyte(0x0756)
    local status_diff = 0
    if prev_status ~= -1 then
        status_diff = status - prev_status
    end
    prev_status = status

    -- Advancement rank
    local world = memory.readbyte(0x075C)
    local level = memory.readbyte(0x0760)
    local rank = world * 10 + level
    local rank_diff = 0
    if rank > prev_rank then
        rank_diff = rank - prev_rank
    end
    prev_rank = rank

    local position = memory.readbyte(0x071d)
    local position_diff = 0
    if position ~= prev_position then
        prev_position = position
        -- Advances sporadically & rolls over often but never backwards
        position_diff = 1
        no_advance_timer = 0
    else
        no_advance_timer = no_advance_timer + 1
    end
    local no_advance_time = 0
    if no_advance_timer > 16 then
        -- Intensity should increase gradually,(non-linearly)
        -- so we use a square root function.
        no_advance_time = math.sqrt(no_advance_timer - 16)
    end

    -- Score
    local points =
        memory.readbyte(0x07DD) * 1000000 + 
        memory.readbyte(0x07DE) * 100000 + 
        memory.readbyte(0x07DF) * 10000 +
        memory.readbyte(0x07E0) * 1000 +
        memory.readbyte(0x07E1) * 100 +
        memory.readbyte(0x07E2) * 10
    local points_diff = points - prev_points
    prev_points = points

    -- Star timer
    local star_timer = memory.readbyte(0x079F)
    local star_timer_diff = 0
    if prev_star_timer == 0 and star_timer > 0 then
        star_timer_diff = 1
    end
    prev_star_timer = star_timer

    local reward = rank_diff * ADVANCE_REWARD +
        game_over * -ADVANCE_REWARD +
        life_diff * LIFE_REWARD +
        status_diff * POWER_REWARD +
        star_timer_diff * POWER_REWARD +
        coin_diff * COIN_REWARD +
        points_diff * POINT_REWARD +
        position_diff * STEP_REWARD
        --no_advance_time * -NO_STEP_PENALTY

--  if reward ~= 0 then
--        print('REWARD:')
--        print("coins: " .. coins .. " versus previous " .. prev_coins)
--        print("lives: " .. lives .. " versus previous " .. prev_lives)
--        print("status: " .. status .. " versus previous " .. prev_status)
--        print("rank: " .. rank .. " versus previous " .. prev_rank)
--        print("points: " .. points .. " versus previous " .. prev_points)
--        print("position: " .. position .. " versus previous " .. prev_position)
--        print("star timer: " .. star_timer_diff .. " versus previous " .. prev_star_timer)
--        print("paused: " .. paused .. " versus previous " .. prev_paused)
--        print("TOTAL: " .. reward)
--    end

    return reward
end

-- Main emulation loop
while true do
    frame = emu.framecount()
    writeScreenToFile()
    reward = getRewardSMB()
    if reward == nil then
        pipe:write("     nan")
    else
        print('frame ' .. frame .. ' reward ' .. reward)
        pipe:write(string.format("%8.3f", reward))
    end
    emu.frameadvance()
    writeInputToFile()
    pipe:flush()
end

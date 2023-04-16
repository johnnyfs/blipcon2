-- Writes raw states to a named file (presumably a pipe),
-- and reads raw actions from another file. If a game code
-- is specified and recognized, then custom logic will be
-- used to determine the reward for the state. Otherwise,
-- the reward will be NaN. (Signaling that a novelty-based
-- reward should be used by the predictor instead.)
local infilename = args[1]   -- required
local outfilename = args[2]  -- required
local game = 'undefined'
if args:len() > 2 then
    game = args[3]
end

local ADVANCE_REWARD = 100.0 -- also game over penalty
local LIFE_REWARD = 10.0     -- for gain or loss of 1 life
local POWER_REWARD = 5.0     -- status up or down or starman
local COIN_REWARD = 1.0      -- for gain of 1 coin
local POINT_REWARD = 0.001   -- 1 reward every 1000 points 

local BUTTONS= {
    "A",
    "B",
    "Select",
    "Start",
    "Up",
    "Down",
    "Left",
    "Right"
}

local infile = io.open(infilename, "rb")
local outfile = io.open(outfilename, "wb")

local prev_coins = 101
local prev_lives = -1
local prev_status = -1
local prev_rank = 256
local prev_points = 0
local prev_star_timer = 0

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

    -- Score
    local points =
        memory.readbyte(0x07DD) * 1000000 + 
        memory.readbyte(0x07DE) * 100000 + 
        memory.readbyte(0x07DF) * 10000 +
        memory.readbyte(0x07E0) * 1000 +
        memory.readbyte(0x07E1) * 100 +
        memory.readbyte(0x07E2) * 10
    local points_diff = points - prev_points

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
        points_diff * POINT_REWARD

    return reward
end

function getReward()
    if game == 'smb' then
        return getRewardSMB()
    else
        return nil
    end
end

function getState()
    return gui.gdscreenshot()
end

function postToPredict(state, rewward)
    -- Write the state to the output pipe
    outfile:write(state)
    -- Write the reward as a 0-padded string representation
    -- of the floating point value of exactly 8 characters
    if reward == nil then
        outfile:write("    nan")
    else
        outfile:write(string.format("%8.3f", reward))
    end

    -- Read the prediction as an 8 1-character representations
    -- of the button states (0 = false, 1 = true)
    -- return a table of the states
    local prediction = infile:read(8)
    local actions = {}
    for i = 1, 8 do
        if prediction:sub(i, i) == "1" then
            button = BUTTONS[i]
            actions[button] = true
        end
    end
    return actions
end

-- Main emulation loop
while true do
    local state = getState()   -- Resulting from last action or first state
    local reward = getReward() -- Resulting from last action or 0
    local actions = postToPredict(state, reward)
    joypad.set(1, actions)
    emu.frameadvance()
end
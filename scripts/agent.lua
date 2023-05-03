-- Writes raw states to a named file (presumably a pipe),
-- and reads raw actions from another file. If a game code
-- is specified and recognized, then custom logic will be
-- used to determine the reward for the state. Otherwise,
-- the reward will be NaN. (Signaling that a novelty-based
-- reward should be used by the predictor instead.)

function split(inputstr, sep)
    sep=sep or '%s'
    local t={}
    for field,s in string.gmatch(inputstr, "([^"..sep.."]*)("..sep.."?)") do
        print(field)
        print(s)
        table.insert(t,field)
        if s=="" then
            return t
        end
    end
    return t
end

args = split(arg, " ")
local outfilename = args[1]  -- required
local infilename = args[2]   -- required
local game = args[3]
local free_play = args[4]

print("infilename: " .. infilename)
print("outfilename: " .. outfilename)
print("game: " .. game)

if free_play ~= nil then
    print("free_play: " .. free_play)
    free_play = true
else
    free_play = false
end

local FRAME_SKIP = 0
local ADVANCE_REWARD = 100.0 -- also game over penalty
local LIFE_REWARD = 10.0     -- for gain or loss of 1 life
local POWER_REWARD = 5.0     -- status up or down or starman
local COIN_REWARD = 1.0      -- for gain of 1 coin
local STEP_REWARD = 0.1      -- for advancing within a level
local POINT_REWARD = 0.001   -- 1 reward every 1000 points 
local NO_STEP_PENALTY = 0.01 -- for not advancing (per processed frame)

local BUTTONS= {
    "A",
    "B",
    "select",
    "start",
    "up",
    "down",
    "left",
    "right"
}

print("Opening file" .. outfilename)
local outfile = io.open(outfilename, "wb")
print("Opening file" .. infilename)
local infile = io.open(infilename, "rb")

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

-- Register stop function
function onEmulationExit()
    print("Exiting")
    outfile:close()
    infile:close()
end
emu.registerexit(onEmulationExit)

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
    if no_advance_timer > 256 then
        -- Intensity should increase gradually,(non-linearly)
        -- so we use a square root function.
        no_advance_time = math.sqrt(no_advance_timer - 256)
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

    -- Pause status
    local paused_status = memory.readbyte(0x0776)
    if paused_status == 1 or paused_status == 129 and not free_play then
        -- Hack to make things go faster (auto unpause)
        memory.writebyte(0x0776, 0)
    end

    local reward = rank_diff * ADVANCE_REWARD +
        game_over * -ADVANCE_REWARD +
        life_diff * LIFE_REWARD +
        status_diff * POWER_REWARD +
        star_timer_diff * POWER_REWARD +
        coin_diff * COIN_REWARD +
        points_diff * POINT_REWARD +
        position_diff * STEP_REWARD +
        no_advance_time * -NO_STEP_PENALTY

  if reward ~= 0 then
        print('REWARD:')
        print("coins: " .. coins .. " versus previous " .. prev_coins)
        print("lives: " .. lives .. " versus previous " .. prev_lives)
        print("status: " .. status .. " versus previous " .. prev_status)
        print("rank: " .. rank .. " versus previous " .. prev_rank)
        print("points: " .. points .. " versus previous " .. prev_points)
        print("position: " .. position .. " versus previous " .. prev_position)
        print("star timer: " .. star_timer_diff .. " versus previous " .. prev_star_timer)
        print("TOTAL: " .. reward)
    end

    return reward
end

function getReward()
    if game == 'smb' then
        return getRewardSMB()
    else
        print('Unknown game: ' .. game)
        return nil
    end
end

function getState()
    return gui.gdscreenshot()
end

function postToPredict(state, reward)
    -- Write the state to the output pipe
    outfile:write(state)
    -- Write the reward as a 0-padded string representation
    -- of the floating point value of exactly 8 characters
    if reward == nil then
        outfile:write("     nan")
    else
        outfile:write(string.format("%8.3f", reward))
    end
    outfile:flush()

    -- Read the prediction as an 8 1-character representations
    -- of the button states (0 = false, 1 = true)
    -- return a table of the states
    -- print("reading predictions")
    if free_play then
        return nil -- skip b/c the response will wait on the override
    end
    local prediction = infile:read(8)
    local actions = {}
    if prediction ~= nil then
        for i = 1, 8 do
            button = BUTTONS[i]
            if prediction:sub(i, i) == "1" then
                actions[button] = true
            else
                actions[button] = false
            end
        end
    else
        print("prediction is nil")
    end
    return actions
end

-- Main emulation loop
while true do
    local frame = emu.framecount()
    if frame - last_frame - 1 < FRAME_SKIP then
        if last_actions ~= nil then
            joypad.set(1, last_actions)
            joypad.write(1, last_actions)
        end
        emu.frameadvance()
    else
        last_frame = frame
        local state = getState()   -- Resulting from last action or first state
        local reward = getReward() -- Resulting from last action or 0
        print("frame " .. frame .. " reward " .. reward)
        local actions = {}
        if first then
            actions['start'] = true
        else
            actions = postToPredict(state, reward)
        end

        if free_play and not first then
            actions = joypad.get(1)
            for i = 1, 8 do
                button = BUTTONS[i]
                print(button .. ": " .. tostring(actions[button]))
                if actions[button] then
                    outfile:write("1")
                else
                    outfile:write("0")
                end
            end
            outfile:flush()
            __ = infile:read(8) -- ignore the response
        else
            joypad.set(1, actions)
        end
        emu.frameadvance()
        last_actions = actions
        first = false
    end
end
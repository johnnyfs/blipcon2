print("starting!")
print("testing " .. arg)

function display_values()
    local coins = memory.readbyte(0x075E) * 1 + memory.readbyte(0x075F) * 10
    local time_remaining = memory.readbyte(0x07F8) * 100 + memory.readbyte(0x07F9) * 10 + memory.readbyte(0x07FA)
    local lives = memory.readbyte(0x075A)
    local points =
        memory.readbyte(0x07DD) * 1000000 + 
        memory.readbyte(0x07DE) * 100000 + 
        memory.readbyte(0x07DF) * 10000 +
        memory.readbyte(0x07E0) * 1000 +
        memory.readbyte(0x07E1) * 100 +
        memory.readbyte(0x07E2) * 10
    local world = memory.readbyte(0x075C)
    local level = memory.readbyte(0x0760)
    local mario_status = memory.readbyte(0x0756)
    local mario_statuses = {"Small", "Big", "Fiery"}
    local invincibility_timer = memory.readbyte(0x079F)
    local level_end = memory.readbyte(0x07B8)

    print("Coins: " .. coins)
    print("Time Remaining: " .. time_remaining)
    print("Lives: " .. lives)
    print("Points: " .. points)
    print("World: " .. world)
    print("Level: " .. level)
    print("Mario Status: " .. mario_statuses[mario_status + 1])
    print("Invincibility Timer: " .. invincibility_timer)
    print("Level End: " .. level_end)
    print("---")
end

while true do
    -- display_values()
    emu.frameadvance()
end
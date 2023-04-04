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

-- Main emulation loop
while true do
    writeScreenToFile()
    emu.frameadvance()
    writeInputToFile()
end

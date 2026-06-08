# CONTRIBUTING

This project is lightweight and should build with a standard Rust toolchain and system OpenGL support.

For basic usage and examples, refer to the README.

## Requirements
This project uses OpenGL drivers, and the usual GLFW and ImGui dependencies handled through the build.

## Build Instructions
Clone the repo and run:  
`cargo build`  
`cargo run`

To run this with one of the included example files use:  
`cargo run -- examples/pikachu.craft`

## Platform Notes

### Windows
WSL2 with Ubuntu is probably the most stable option. Native builds may require MSVC or MinGW, working OpenGL drivers, and sometimes binutils for windres.

If you encounter (so far only tested with MinGW):  
`Error: RC or WINDRES should be defined`

you can fix it by manually setting the a variable WINDRES:  
`set WINDRES=C:\msys64\mingw64\bin\windres.exe`

this of course this works only if binutils is installed (which it normally should be the case):  
`pacman -S mingw-w64-x86_64-binutils`

## Submitting Changes

* Please open an issue to discuss what you want to implement before starting to work on it
* Use feature branches such as `feature/x` or `fix/y`
* Run `cargo fmt` before submitting
* Describe clearly what changed and why in your pull request
* Include screenshots if your changes affect the UI
* Preferably refrain from excessive swearing in your code

## Notes

This is a hobby project and not a tightly governed codebase. Improvements to portability and build reliability are welcome. If it fails to build on your system, open an issue with full logs and system details.
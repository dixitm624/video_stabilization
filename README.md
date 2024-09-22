# Video Stabilization Using OpenCV

This repository contains a C++ implementation of **video stabilization** using OpenCV. The program utilizes input from a webcam to analyze camera motion through feature matching across the first **100 frames**. The calculated transformations are then applied to subsequent frames (from the **101st to the 201st frame**), resulting in a stabilized video output.

**Note:** This implementation is not real-time, as transformations are applied based on previously calculated values.

## Requirements

- **Operating System:** Linux (Ubuntu recommended)
- **OpenCV:** Version 3.4 or higher
- **CMake:** Version 3.10 or higher
- **g++:** Version 7 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/video_stabilization.git
   cd video_stabilization
2. **Create a build directory**
    `mkdir build` 
    `cd build`
3. **Compile the code**
    `cmake ..`
    `make`

4. **Run the exe**
    `./video_stabilization`
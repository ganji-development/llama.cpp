$rock       = 'E:\TheRock\build'
$llvm       = 'C:\Program Files\LLVM'
$openssl    = 'C:\Program Files\OpenSSL-Win64'
$opensslLib = 'C:\Program Files\OpenSSL-Win64\lib\VC\x64\MD'
$opensslCmakeRoot = "$PWD\.deps\openssl-win64"
$ninja      = 'C:\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe'

Remove-Item -Recurse -Force .\.deps\openssl-win64 -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force "$opensslCmakeRoot\lib" | Out-Null
Copy-Item "$openssl\include" "$opensslCmakeRoot\include" -Recurse -Force
Copy-Item "$opensslLib\libcrypto.lib" "$opensslCmakeRoot\lib\libcrypto.lib" -Force
Copy-Item "$opensslLib\libssl.lib"    "$opensslCmakeRoot\lib\libssl.lib"    -Force

$env:HIP_PATH = $rock
$env:ROCM_PATH = $rock
$env:ROCBLAS_TENSILE_LIBPATH = "$rock\bin\rocblas\library"
$env:PATH = "$rock\lib\llvm\bin;$rock\bin;$llvm\bin;$openssl\bin;$env:PATH"

Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue

cmake -S . -B .\build -G Ninja `
  -DCMAKE_MAKE_PROGRAM="$ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$rock/lib/llvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$rock/lib/llvm/bin/clang++.exe" `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows;$rock" `
  -DGGML_HIP=ON `
  -DGPU_TARGETS=gfx1100 `
  -DOpenMP_C_FLAGS="-fopenmp" `
  -DOpenMP_CXX_FLAGS="-fopenmp" `
  -DOpenMP_C_LIB_NAMES="omp" `
  -DOpenMP_CXX_LIB_NAMES="omp" `
  -DOpenMP_omp_LIBRARY:FILEPATH="$llvm/lib/libomp.lib" `
  -DOPENSSL_ROOT_DIR:PATH="$opensslCmakeRoot" `
  -DOPENSSL_USE_STATIC_LIBS=FALSE `
  -DCMAKE_C_FLAGS="--rocm-device-lib-path=$rock/lib/llvm/amdgcn/bitcode" `
  -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=$rock/lib/llvm/amdgcn/bitcode"

cmake --build .\build --parallel
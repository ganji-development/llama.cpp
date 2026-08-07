# Every dependency comes from this machine's own trees: the natively built ROCm
# supplies clang, the device bitcode, the HIP/rocBLAS runtime and OpenMP, and
# the external/openssl submodule supplies TLS. Nothing is taken from
# C:\Program Files, so the build does not depend on what is installed system-wide.
$rock       = 'E:\rocm\build_gfx1100\dist\rocm'
$rockLlvm   = "$rock\lib\llvm"
$openssl    = Join-Path (Split-Path $PSScriptRoot -Parent) 'openssl'
$opensslCmakeRoot = "$PWD\.deps\openssl-win64"
$ninja      = 'C:\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe'

if (-not (Test-Path "$openssl\libssl.lib")) {
    throw "external/openssl is not built: no libssl.lib in $openssl"
}

Remove-Item -Recurse -Force .\.deps\openssl-win64 -ErrorAction SilentlyContinue

# OpenSSL builds in place, leaving its headers, import libraries and DLLs all at
# the top of its source tree. FindOpenSSL wants the usual include/ + lib/ + bin/
# layout, so mirror it into one; the DLLs come along because the staging rule in
# vendor/cpp-httplib looks for them relative to the import library.
New-Item -ItemType Directory -Force "$opensslCmakeRoot\lib" | Out-Null
New-Item -ItemType Directory -Force "$opensslCmakeRoot\bin" | Out-Null
Copy-Item "$openssl\include" "$opensslCmakeRoot\include" -Recurse -Force
Copy-Item "$openssl\libcrypto.lib" "$opensslCmakeRoot\lib\libcrypto.lib" -Force
Copy-Item "$openssl\libssl.lib"    "$opensslCmakeRoot\lib\libssl.lib"    -Force
Copy-Item "$openssl\libcrypto-*-x64.dll" "$opensslCmakeRoot\bin\" -Force
Copy-Item "$openssl\libssl-*-x64.dll"    "$opensslCmakeRoot\bin\" -Force

$env:HIP_PATH = $rock
$env:ROCM_PATH = $rock
$env:ROCBLAS_TENSILE_LIBPATH = "$rock\bin\rocblas\library"
$env:PATH = "$rockLlvm\bin;$rock\bin;$openssl;$env:PATH"

Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue

# -fno-cuda-host-device-constexpr is what lets the HIP sources compile against
# MSVC 14.51. Its <cmath> declares isgreater/isless/... `constexpr` (14.44 had
# them `inline`), and clang-HIP promotes unmarked constexpr functions to
# implicit __host__ __device__, which then collides with clang's own __device__
# forward declares in __clang_cuda_math_forward_declares.h. Turning the
# promotion off is enough; nothing in the ROCm tree needs patching.
cmake -S . -B .\build -G Ninja `
  -DCMAKE_MAKE_PROGRAM="$ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$rockLlvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$rockLlvm/bin/clang++.exe" `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows;$rock" `
  -DGGML_HIP=ON `
  -DGPU_TARGETS=gfx1100 `
  -DOpenMP_C_FLAGS="-fopenmp" `
  -DOpenMP_CXX_FLAGS="-fopenmp" `
  -DOpenMP_C_LIB_NAMES="omp" `
  -DOpenMP_CXX_LIB_NAMES="omp" `
  -DOpenMP_omp_LIBRARY:FILEPATH="$rockLlvm/lib/libomp.lib" `
  -DOPENSSL_ROOT_DIR:PATH="$opensslCmakeRoot" `
  -DOPENSSL_USE_STATIC_LIBS=FALSE `
  -DCMAKE_C_FLAGS="--rocm-device-lib-path=$rockLlvm/amdgcn/bitcode" `
  -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=$rockLlvm/amdgcn/bitcode -Xclang -fno-cuda-host-device-constexpr"

cmake --build .\build --parallel
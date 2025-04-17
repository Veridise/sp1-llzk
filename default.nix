{
  # nix utils
  stdenv, lib, rustPlatform,

  # dependencies
  mlir, llzk,
  cmake, cargo, llvmPackages_18,
  ncurses, xml2
}:
rustPlatform.buildRustPackage rec {
  pname = "sp1-llzk";
  version = "0.1.0";
            
  src = ./.;
  cargoLock = {
    lockFile = ./Cargo.lock;
  };
  nativeBuildInputs = [ cmake cargo clang llvmPackages_18.llvm mlir.dev xml2 ];
  buildInputs = [ llzk mlir.dev ncurses xml2 llvmPackages_18.libllvm.dev ];

  #LIBCLANG_PATH = "${pkgs.llvmPackages.libclang}/lib";
  MLIR_PATH = "${mlir.lib}/lib";
  LLZK_PATH = "${llzk}/lib";

  preBuild = ''
    # From: https://github.com/NixOS/nixpkgs/blob/1fab95f5190d087e66a3502481e34e15d62090aa/pkgs/applications/networking/browsers/firefox/common.nix#L247-L253
    # Set C flags for Rust's bindgen program. Unlike ordinary C
    # compilation, bindgen does not invoke $CC directly. Instead it
    # uses LLVM's libclang. To make sure all necessary flags are
    # included we need to look in a few places.
    export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
      $(< ${stdenv.cc}/nix-support/libc-cflags) \
      $(< ${stdenv.cc}/nix-support/cc-cflags) \
      $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
      ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
      ${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"} \
    "
  '';

  # This is done specifically so that the configure phase can find /usr/bin/sw_vers,
  # which is MacOS specific.
  # Note that it's important for "/usr/bin/" to be last in the list so we don't
  # accidentally use the system clang, etc.
  preConfigure = ''
    if [[ "$(uname)" == "Darwin" ]]; then
      export OLD_PATH=$PATH
      export PATH="$PATH:/usr/bin/"
    fi
  '';

  # this undoes the above configuration, as it will cause problems later.
  postConfigure = ''
    if [[ "$(uname)" == "Darwin" ]]; then
      export PATH=$OLD_PATH
    fi
  '';

  doCheck = false;
};


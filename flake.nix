{
  inputs = {
    llzk-pkgs.url = "github:Veridise/llzk-nix-pkgs?ref=main";

    nixpkgs = {
      url = "github:NixOS/nixpkgs";
      follows = "llzk-pkgs/nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils/v1.0.0";
      follows = "llzk-pkgs/flake-utils";
    };

    llzk = {
      url = "git+ssh://git@github.com/Veridise/llzk-lib.git?ref=main";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, llzk-pkgs, llzk }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import nixpkgs) {
          inherit system; 
          overlays = [
            llzk-pkgs.overlays.default
            llzk.overlays.default
          ];
        };
      in rec {
        packages = {
          default = with pkgs; pkgs.rustPlatform.buildRustPackage rec {
            pname = "sp1-llzk";
            version = "0.1.0";
            
            #LIBCLANG_PATH = "${pkgs.llvmPackages.libclang}/lib";
            doCheck = false;

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
        # unset OLD_PATH
      fi
    '';

            nativeBuildInputs = [ cmake cargo clang llvmPackages_18.llvm mlir.dev ];
            buildInputs = [ pkgs.llzk mlir ncurses llvmPackages_18.libllvm.dev ];
            src = ./.;
            cargoLock = {
              lockFile = ./Cargo.lock;
            };
          };
        };

        devShells = with pkgs; mkShell {
          buildInputs = [ cargo rustc rustfmt pre-commit rustPackages.clippy llzk ];
          #RUST_SRC_PATH = rustPlatform.rustLibSrc;
        };
      });
}

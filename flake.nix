{
  inputs = {
    # This repo is public already
    llzk-pkgs.url = "github:Veridise/llzk-nix-pkgs?ref=main";

    nixpkgs = {
      url = "github:NixOS/nixpkgs";
      follows = "llzk-pkgs/nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils/v1.0.0";
      follows = "llzk-pkgs/flake-utils";
    };

    # LLZK is not open-source (yet)
    # llzk = {
    #   url = "git+ssh://git@github.com/Veridise/llzk-lib.git?ref=dani/LLZK-253-LLZK-to-Picus-translation";
    #   inputs.nixpkgs.follows = "nixpkgs";
    #   inputs.llzk-pkgs.follows = "llzk-pkgs";
    # };
  };

  outputs = { self, nixpkgs, flake-utils, llzk-pkgs
    #, llzk 
  }:
    {
      overlays.default = final: prev: {
        sp1-llzk = final.callPackage ./. { clang = final.clang_18; 
          #llzk = final.llzk; 
        };
      };
    }//
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system; 
          overlays = [
            self.overlays.default
            llzk-pkgs.overlays.default
            #llzk.overlays.default
          ];
        };
      in {
        packages = {
          inherit (pkgs) sp1-llzk;
          default = pkgs.sp1-llzk;
        };

        devShells = flake-utils.lib.flattenTree {
          default = pkgs.sp1-llzk.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              rustfmt 
              rustPackages.clippy 
            ]);

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.z3.lib}/lib:$LD_LIBRARY_PATH
            '';
          });
        };
      }
    ));
}

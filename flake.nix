{
  inputs = {
    naersk.url = "github:nix-community/naersk/master";
    utils.url = "github:numtide/flake-utils/v1.0.0";
  };

  outputs = { self, nixpkgs, utils, naersk }:

    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        naersk-lib = pkgs.callPackage naersk { };
      in rec {
        defaultPackage = with pkgs; naersk-lib.buildPackage {
           name = "sp1-llzk";
           pname = "sp1-llzk";
           src = ./.;
         };

        devShells = with pkgs; mkShell {
          buildInputs = [ cargo rustc rustfmt pre-commit rustPackages.clippy ];
          RUST_SRC_PATH = rustPlatform.rustLibSrc;
        };
      });
}

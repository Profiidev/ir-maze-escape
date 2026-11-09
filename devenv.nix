{ pkgs, ... }:

{
  languages.python = {
    enable = true;
    package = pkgs.python314;
    uv.enable = true;
    # Wheels (numpy, matplotlib) need these on NixOS.
    libraries = [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      # matplotlib dlopens these to detect a display; without them it falls back to Agg.
      pkgs.libx11
      pkgs.wayland
    ];
  };

  # tkinter is a separate package in nixpkgs; matplotlib needs it for plt.show().
  # Appended, not set: devenv already uses PYTHONPATH for its sitecustomize.
  enterShell = ''
    export PYTHONPATH="$PYTHONPATH:${pkgs.python314.pkgs.tkinter}/${pkgs.python314.sitePackages}"
  '';
}

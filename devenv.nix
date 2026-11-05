{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  languages.python.enable = true;
  languages.python.uv.enable = true;

  packages = with pkgs; [zlib jupyter];

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/venv" ]; then
      ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/venv"
    fi
  '';
}

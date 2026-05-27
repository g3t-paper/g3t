import rich

_log_styles = {
    "VGGT-Long": "bold green",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"

def Log(*args, tag="VGGT-Long"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)

try:
    # Works for `python -m drone_geofence`.
    from .app import main
except ImportError:
    # Works when this file is executed as a direct script by build tools.
    from drone_geofence.app import main

if __name__ == "__main__":
    main()

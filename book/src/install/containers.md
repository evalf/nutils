# Using Containers

As an alternative to installing Nutils, it is possible to download a
preinstalled system image with all the above considerations taken care of.
Nutils provides [OCI](https://opencontainers.org) compatible containers for all
releases, as well as the current developement version, which can be run using
tools such as [Docker](https://www.docker.com) or [Podman](https://podman.io).
The images are hosted in [Github's container
repository](https://github.com/evalf/nutils/pkgs/container/nutils).

The container images include all the :ref:`examples`. To run one, add the name
of the example and any additional arguments to the command line. For example,
you can run example `laplace` using the latest version of Nutils with:

```sh
docker run --rm -it ghcr.io/evalf/nutils:latest laplace
```

HTML log files are generated in the `/log` directory of the container. If you
want to store the log files in `/path/to/log` on the host, add `-v
/path/to/log:/log` to the command line before the name of the image. Extending
the previous example:

```sh
docker run --rm -it -v /path/to/log:/log ghcr.io/evalf/nutils:latest laplace
```

To run a Python script in this container, bind mount the directory containing
the script, including all files necessary to run the script, to `/app` in the
container and add the relative path to the script and any arguments to the
command line. For example, you can run `/path/to/myscript.py` with Docker
using:

```sh
docker run --rm -it -v /path/to:/app:ro ghcr.io/evalf/nutils:latest myscript.py
```


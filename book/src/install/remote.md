# Remote Computing

Computations beyond a certain size are usually moved to a remote computing
facility, typically accessed using tools such as [Secure
Shell](https://en.wikipedia.org/wiki/Secure_Shell) or [Mosh](https://mosh.org),
combined with a terminal multiplexer such as [GNU
Screen](https://www.gnu.org/software/screen/) or
[Tmux](https://github.com/tmux/tmux/wiki). In this scenario it is useful to
install a webserver for remote viewing of the html logs.

The standard `~/public_html` output directory is configured with the scenario
in mind, as the [Apache](https://httpd.apache.org/) webserver uses this as the
default [user
directory](https://httpd.apache.org/docs/2.4/howto/public_html.html). As this
is disabled by default, the module needs to be enabled by editing the relevant
configuration file or, in Debian Linux, by using the `a2enmod` utility::

```sh
sudo a2enmod userdir
```

Similar behaviour can be achieved with the [Nginx](https://www.nginx.com/) by
configuring a location pattern in the appropriate server block:

```
location ~ ^/~(.+?)(/.*)?$ {
  alias /home/$1/public_html$2;
}
```

Finally, the terminal output can be made to show the http address rather than
the local uri by adding the following line to the `~/.nutilsrc` configuration
file:

```
outrooturi = 'https://mydomain.tld/~myusername/'
```

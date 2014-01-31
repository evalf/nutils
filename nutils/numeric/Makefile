MODULE=_numeric

$(MODULE).so: $(MODULE).c
	python setup.py build
	cp build/lib*/$@ .
	rm -r -f build

$(MODULE)_d.so: $(MODULE).c
	python-dbg setup.py build
	cp build/lib*/$@ .
	rm -r -f build

clean:
	rm -r -f build $(MODULE).so $(MODULE)_d.so

.PHONY: clean

# vim:noexpandtab

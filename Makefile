MODULE=_numeric

$(MODULE).so: $(MODULE).c clean
	python setup.py build
	cp build/lib*/$(MODULE).so .

$(MODULE)_d.so: $(MODULE).c clean
	python-dbg setup.py build
	cp build/lib*/$(MODULE)_d.so .

clean:
	rm -r -f build

.PHONY: clean

# vim:noexpandtab

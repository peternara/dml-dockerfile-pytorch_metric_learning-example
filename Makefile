run:
	nvidia-docker run -it -v $(CURDIR):/work/ -v /media/data/oike/:/media/ --name oike-dml toshi17/deep-metric-learning
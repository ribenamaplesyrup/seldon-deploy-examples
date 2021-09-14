# Deploy Retinanet with Seldon

## To Get Started ...

1. ``git clone`` this repo.
2. Open your command prompt and navigate to `seldon-deploy-examples/retinanet-pytorch/model`
3. Create and activate an environment with 'torch' and 'torchvision' installed.
4. Run `test.py` to download the model from Torch Hub and run inference on a test image.
5. Create `mnt/models` and copy the downloaded model to this location. 
6. Build a container image ``s2i build . seldonio/seldon-core-s2i-python3:1.10.0-dev <my-image-name>``
    ie ``s2i build . seldonio/seldon-core-s2i-python3:1.10.0-dev retinanet:0.1``
7. Check container image runs: ``docker run --name "retinanet" -p 9001:9000 retinanet:0.1``
8. Navigate to `seldon-deploy-examples/retinanet-pytorch` and test running container with `test.py`
9. Commit container: ``docker commit <containerID> retinanet``
10. Tag image: ``docker tag retinanet seangreaves/retinanet:0.1``
11. Push container to repo: ``docker push seangreaves/retinanet:0.1``
12. Deploy to Seldon making sure to allocate enough memory to the deployment (2GB+)
13. Test the deployment with test image at `test.json`

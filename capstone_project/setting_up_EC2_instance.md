
# Setting up an EC2 instance for Deep Learning

This "how to" assumes you have an AWS account and don't mind paying a small amount to set up an EC2 instance. The ML engineer nanodegree with Udacity came with £100 of credits which were used here to train the model.

### Creating an EC2 instance

1. Services -> EC2 -> Launch Instance -> AWS Marketplace ->  Deep Learning AMI with Source Code (CUDA 8, Ubuntu) 
2. Select instance type `p2.xlarge` -> Review and Launch
3. Select the keyPair to use if you already have one, otherwise create one. 
4. The instance will now take a few minutes to launch. 
5. Add the custom security group as shown below to be able to access to Jupyter notebook instance via the browser.

### Connecting to the EC2 instance

1. Ensure you have access to the KeyPair file `chmod 400 MyKeyPair.pem`
2. `ssh -i "MyKeyPair.pem" ubuntu@ec{X.X.X.X}.eu-west-1.compute.amazonaws.com`. This can be found in the Public DNS section (IPV4) and easily copied to the clipboard.

### Create the Jupyter Notebook

1. `jupyter notebook --generate-config`
2. `sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py`
3. `jupyter notebook --ip=0.0.0.0 --no-browser`
4. Copy the URL outputted in the terminal and add the public IP address (IPv4 Public IP
) and then head over to the browser.  
5. It should look something like this: http://65.33.187.41:8888/?token=1d3db0c06476dc55c7569c8fb705ab2053f560cb9b810fae

### Copy over files from local computer download Filezilla [...](https://angus.readthedocs.io/en/2014/amazon/transfer-files-between-instance.html)

1. Download [FileZilla](https://filezilla-project.org/)
2. Edit -> Settings -> Select SFTP
3. Add keyPair.pem file.
4. Create a new site.

### Improvements
* There is a need to reconnec to the EC2 instance and run `jupyter notebook --ip=0.0.0.0 --no-browser` each time which could likely be automated when the EC2 starts.

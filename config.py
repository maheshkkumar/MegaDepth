from yacs.config import CfgNode as CN

# config
_C = CN()

_C.display_freq = 100
_C.print_freq = 100
_C.save_latest_freq = 5000
_C.save_epoch_freq = 5
_C.continue_train = False
_C.phase = 'train'
_C.which_epoch = 'latest'
_C.niter = 100
_C.niter_decay = 100
_C.beta1 = 0.5
_C.lr = 0.0002
_C.no_lsgan = False
_C.lambda_A = 10.0
_C.lambda_B = 10.0
_C.pool_size = 50
_C.no_html = False
_C.no_flip = False
_C.isTrain = True

_C.batchSize = 1
_C.loadSize = 286
_C.fineSize = 256
_C.input_nc = 3
_C.output_nc = 3
_C.ngf = 64
_C.ndf = 64
_C.which_model_netG = 'unet_256'
_C.gpu_ids = [0]
_C.name = 'weights'
_C.model = 'pix2pix'
_C.nThreads = 2
_C.checkpoints_dir = './models/MegaDepth/'
_C.norm = 'instance'
_C.serial_batches = False
_C.display_winsize = 256
_C.display_id = 1
_C.identity = 0.0
_C.no_dropout = False
_C.max_dataset_size = float("inf")
_C.initialized = True

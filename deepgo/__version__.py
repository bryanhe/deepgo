""" Generated automatically---don't edit!
"""

__version__ = '0.0.0+untagged.b97e9c6.dirty'
__diff__ = """diff --git a/histonet/__init__.py b/histonet/__init__.py
index f2e97e8..f0874a3 100644
--- a/histonet/__init__.py
+++ b/histonet/__init__.py
@@ -1,4 +1,4 @@
-# WARNING: get_parser must be called very early on for argcomplete.
+# WARNING: parser must be called very early on for argcomplete.
 # argcomplete evaluates the package until the parser is constructed before it
 # can generate completions. Because of this, the parser must be constructed
 # before the full package is imported to behave in a usable way. Note that
@@ -8,8 +8,9 @@
 # pytorch, numpy, and pandas), before running __main__.py, which takes
 # about 0.5-1 seconds.
 # See Performance section of https://argcomplete.readthedocs.io/en/latest/
-from histonet.parser import get_parser
-get_parser()
+
+from .parser import parser
+parser()
 
 from histonet.__version__ import __version__
 from histonet.config import config
@@ -19,4 +20,4 @@ import histonet.datasets as datasets
 import histonet.cmd as cmd
 import histonet.models as models
 import histonet.utils as utils
-import histonet.transforms as transforms
+import histonet.transforms as transforms
\ No newline at end of file
diff --git a/histonet/cmd/prepare/__init__.py b/histonet/cmd/prepare/__init__.py
index e69de29..7c722c2 100644
--- a/histonet/cmd/prepare/__init__.py
+++ b/histonet/cmd/prepare/__init__.py
@@ -0,0 +1 @@
+from .spatial import spatial
\ No newline at end of file
diff --git a/histonet/cmd/prepare/spatial.py b/histonet/cmd/prepare/spatial.py
index a1d9df5..12ce769 100644
--- a/histonet/cmd/prepare/spatial.py
+++ b/histonet/cmd/prepare/spatial.py
@@ -24,7 +24,7 @@ import histonet.utils.util as util
 import histonet.utils.logging
 
 
-def prepare(args):
+def spatial(args):
 
     window = 224  # only to check if patch is off of boundary
     histonet.utils.logging.setup_logging(args.logfile, args.loglevel)
diff --git a/histonet/cmd/train_classifier.py b/histonet/cmd/train_classifier.py
index a768582..d104d63 100644
--- a/histonet/cmd/train_classifier.py
+++ b/histonet/cmd/train_classifier.py
@@ -1,5 +1,3 @@
-#!/usr/bin/env python
-
 import torch
 import torch.nn.functional
 import numpy as np
@@ -39,18 +37,20 @@ def train(args=None):
         torch.manual_seed(args.seed)
 
         logger.info(args)
-        try:
+        logger.info(\"Configuration file: {}\".format(histonet.config.FILENAME))
+        if \"CUDA_VISIBLE_DEVICES\" in os.environ:
             logger.info(\"CUDA_VISIBLE_DEVICES: {}\".format(os.environ[\"CUDA_VISIBLE_DEVICES\"]))
-        except KeyError:
+        else:
             logger.info(\"CUDA_VISIBLE_DEVICES not defined.\")
-
         logger.info(\"CPUs: {}\".format(os.sched_getaffinity(0)))
         logger.info(\"GPUs: {}\".format(torch.cuda.device_count()))
         logger.info(\"Hostname: {}\".format(socket.gethostname()))
 
+        device = (\"cuda\" if args.gpu else \"cpu\")
 
         # Pretrained Model
-        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
+        model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=123)
+        print(model)
 
         if args.dataset == \"TCGA\":
             patient = list(map(lambda x: x.split(\"/\")[-1].split(\".\")[0], glob.glob(\".tcga/brca/*.npz\")))
@@ -66,6 +66,7 @@ def train(args=None):
             train_patients = []
             test_patients = []
 
+            print(patient)
             for p in patient:
                 if args.trainpatients is None and args.testpatients is None:
                     # Random allocation
@@ -161,7 +162,7 @@ def train(args=None):
 
         if args.gpu:
             model = torch.nn.DataParallel(model)
-            model.cuda()
+        model.to(device)
 
         if args.load is not None:
             # TODO: needs some changes for checkpoint with optim
@@ -183,8 +184,7 @@ def train(args=None):
             else:
                 histonet.utils.nn.set_out_features(model, outputs)
 
-        if args.gpu:
-            model.cuda()
+        model.to(device)
 
         if args.finetune is None:
             parameters = model.parameters()
@@ -240,15 +240,16 @@ def train(args=None):
             mean_expression_tumor /= float(tumor)
             mean_expression_normal /= float(normal)
             median_expression = torch.log(1 + torch.Tensor(train_dataset.median_expression))
-            if args.gpu:
-                mean_expression = mean_expression.cuda()
-                mean_expression_tumor = mean_expression_tumor.cuda()
-                mean_expression_normal = mean_expression_normal.cuda()
-                median_expression = median_expression.cuda()
+
+            mean_expression = mean_expression.to(device)
+            mean_expression_tumor = mean_expression_tumor.to(device)
+            mean_expression_normal = mean_expression_normal.to(device)
+            median_expression = median_expression.to(device)
 
             m = model
             if args.gpu:
                 m = m.module
+
             if isinstance(m, torchvision.models.vgg.VGG):
                 last = m.classifier[-1]
             elif isinstance(m, torchvision.models.densenet.DenseNet):
@@ -291,9 +292,10 @@ def train(args=None):
                     for (i, data) in enumerate(loader):
                         X, y, *_ = data
                         y = torch.squeeze(y, dim=1)
-                        if args.gpu:
-                            X = X.cuda()
-                            y = y.cuda()
+
+                        X = X.to(device)
+                        y = y.to(device)
+
                         if args.dataset == \"TCGA\":
                             # TODO: smarter way of doing this?
                             y = (y > 0.5).type(torch.int64)
@@ -354,9 +356,10 @@ def train(args=None):
                 for (i, data) in enumerate(test_loader):
                     X, y, gene, coord, index = data
                     y = torch.squeeze(y, dim=1)
-                    if args.gpu:
-                        X = X.cuda()
-                        y = y.cuda()
+
+                    X = X.to(device)
+                    y = y.to(device)
+
                     pred = model(X)
                     loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')
                     total += loss.cpu().detach().numpy()
@@ -379,8 +382,7 @@ def train(args=None):
 
             if args.gene_mask is not None:
                 args.gene_mask = torch.Tensor([args.gene_mask])
-                if args.gpu:
-                    args.gene_mask = args.gene_mask.cuda()
+                args.gene_mask = args.gene_mask.to(device)
 
             for epoch in range(start_epoch, args.epochs):
                 logger.info(\"Epoch #\" + str(epoch + 1))
@@ -418,10 +420,11 @@ def train(args=None):
                             pixel.append(pix.detach().numpy())
 
                         y = torch.squeeze(y, dim=1)
-                        if args.gpu:
-                            X = X.cuda()
-                            y = y.cuda()
-                            gene = gene.cuda()
+
+                        X = X.to(device)
+                        y = y.to(device)
+                        gene = gene.to(device)
+
                         if dataset == \"test\" and args.average:
                             batch, n_sym, c, h, w = X.shape
                             X = X.view(-1, c, h, w)
@@ -508,6 +511,3 @@ def train(args=None):
     except Exception as e:
         logger.exception(traceback.format_exc())
         raise
-
-if __name__ == \"__main__\":
-    main()
diff --git a/histonet/config.py b/histonet/config.py
index 9597e4e..0ccc08c 100644
--- a/histonet/config.py
+++ b/histonet/config.py
@@ -16,5 +16,6 @@ for filename in [\"histonet.cfg\",
             param = config[\"config\"]
         break
 
-config = types.SimpleNamespace(SPATIAL_RAW_ROOT       = param.get(\"spatial_raw_root\", \"data/hist2tscript/\"),
+config = types.SimpleNamespace(FILENAME = FILENAME,
+                               SPATIAL_RAW_ROOT = param.get(\"spatial_raw_root\", \"data/hist2tscript/\"),
                                SPATIAL_PROCESSED_ROOT = param.get(\"spatial_processed_root\",\"data/hist2tscript/patch/\"))
diff --git a/histonet/main.py b/histonet/main.py
index 95f1150..a82afab 100644
--- a/histonet/main.py
+++ b/histonet/main.py
@@ -7,7 +7,7 @@ import sys
 def main(args=None):
     \"\"\" cli entry point
     \"\"\"
-    parser = histonet.get_parser()
+    parser = histonet.parser()
     args = parser.parse_args(args)
 
     try:
diff --git a/histonet/parser.py b/histonet/parser.py
index 008a06d..593ae6b 100644
--- a/histonet/parser.py
+++ b/histonet/parser.py
@@ -1,8 +1,9 @@
-def get_parser():
-    \"\"\" cli entry point
-    \"\"\"
+import argparse
+
+
+def parser() -> int:
+    \"\"\"Returns parser for histonet.\"\"\"
 
-    import argparse
     import argcomplete
     from datetime import datetime as dt
     from . import __version__
@@ -136,16 +137,18 @@ def get_parser():
     # TODO: multiple datasets
     add_dataset_arguments(classifier_parser)
 
-    # TODO: select train-valid-test partition
     def patient_or_section(name):
         if \"_\" in name:
             return tuple(name.split(\"_\"))
         return name
+
     classifier_parser.add_argument(\"--test\", type=float, default=0.1, help=\"fraction of data as test set\")
-    classifier_parser.add_argument(\"--testpatients\", nargs=\"*\", type=patient_or_section, default=None, help=\"list of data points as test set\"
-                                                                                             \"(--test is ignored if this is set)\")
-    classifier_parser.add_argument(\"--trainpatients\", nargs=\"*\", type=patient_or_section, default=None, help=\"list of data points as train set\"
-                                                                                              \"(defaults to all patients not in test set if not set)\")
+    classifier_parser.add_argument(\"--testpatients\", nargs=\"*\", type=patient_or_section, default=None,
+                                   help=\"list of data points as test set\"
+                                        \"(--test is ignored if this is set)\")
+    classifier_parser.add_argument(\"--trainpatients\", nargs=\"*\", type=patient_or_section, default=None,
+                                   help=\"list of data points as train set\"
+                                        \"(defaults to all patients not in test set if not set)\")
     add_logging_arguments(classifier_parser)
     add_window_arguments(classifier_parser)
     add_device_arguments(classifier_parser)
@@ -153,10 +156,12 @@ def get_parser():
     add_augmentation_arguments(classifier_parser)
 
     group = classifier_parser.add_mutually_exclusive_group()
-    group.add_argument(\"--tumor\", action=\"store_const\", const=\"tumor\", dest=\"task\", default=\"tumor\", help=\"Tumor prediction\")
+    group.add_argument(\"--tumor\", action=\"store_const\", const=\"tumor\", dest=\"task\", default=\"tumor\",
+                       help=\"Tumor prediction\")
     group.add_argument(\"--gene\", action=\"store_const\", const=\"gene\", dest=\"task\", help=\"Gene count prediction\")
     group.add_argument(\"--count\", action=\"store_const\", const=\"count\", dest=\"task\", help=\"Total count prediction\")
-    group.add_argument(\"--geneb\", action=\"store_const\", const=\"geneb\", dest=\"task\", help=\"Gene count binary prediction (high/low)\")
+    group.add_argument(\"--geneb\", action=\"store_const\", const=\"geneb\", dest=\"task\",
+                       help=\"Gene count binary prediction (high/low)\")
 
     add_gene_filter_arguments(classifier_parser)
 
@@ -169,16 +174,22 @@ def get_parser():
                 raise ValueError()
         print([int(i) for i in x])
         return [int(i) for i in x]
-    classifier_parser.add_argument(\"--gene_mask\", type=binary_str, default=None, help=\"binary string with a length matching number of genes\")
-    classifier_parser.add_argument(\"--gene_transform\", dest=\"gene_transform\", choices=[\"none\", \"log\"], default=\"log\", help=\"transform for gene count\")
+
+    classifier_parser.add_argument(\"--gene_mask\", type=binary_str, default=None,
+                                   help=\"binary string with a length matching number of genes\")
+    classifier_parser.add_argument(\"--gene_transform\", dest=\"gene_transform\", choices=[\"none\", \"log\"], default=\"log\",
+                                   help=\"transform for gene count\")
 
     add_normalization_arguments(classifier_parser)
 
-    classifier_parser.add_argument(\"--save_tumor_predictions\", action=\"store_true\", help=\"save tumor predictions to file\")
-    classifier_parser.add_argument(\"--finetune\", type=int, nargs=\"?\", const=1, default=None, help=\"fine tune last n layers\")
+    classifier_parser.add_argument(\"--save_tumor_predictions\", action=\"store_true\",
+                                   help=\"save tumor predictions to file\")
+    classifier_parser.add_argument(\"--finetune\", type=int, nargs=\"?\", const=1, default=None,
+                                   help=\"fine tune last n layers\")
     classifier_parser.add_argument(\"--average\", action=\"store_true\", help=\"average between rotations and reflections\")
 
-    classifier_parser.add_argument(\"--save_genes_every\", type=int, default=None, help=\"how frequently to save gene predictions\")
+    classifier_parser.add_argument(\"--save_genes_every\", type=int, default=None,
+                                   help=\"how frequently to save gene predictions\")
     classifier_parser.add_argument(\"--gene_root\", type=str, default=None, help=\"root for gene prediction outputs\")
     classifier_parser.set_defaults(func=\"histonet.cmd.train_classifier.train\")
 
@@ -203,8 +214,10 @@ def get_parser():
     save_parser.set_defaults(func=\"histonet.cmd.save_patches.save_patches\")
 
     spatial_prep_parser = subparsers.add_parser(\"spatialprep\", help=\"prepare patches for spatial\")
-    spatial_prep_parser.add_argument(\"--root\", type=str, default=config.SPATIAL_RAW_ROOT, help=\"directory containing raw data\")
-    spatial_prep_parser.add_argument(\"--dest\", type=str, default=config.SPATIAL_PROCESSED_ROOT, help=\"destination for patch info\")
+    spatial_prep_parser.add_argument(\"--root\", type=str, default=config.SPATIAL_RAW_ROOT,
+                                     help=\"directory containing raw data\")
+    spatial_prep_parser.add_argument(\"--dest\", type=str, default=config.SPATIAL_PROCESSED_ROOT,
+                                     help=\"destination for patch info\")
     spatial_prep_parser.set_defaults(func=\"histonet.data.spatial.prepare.prepare\")
 
     argcomplete.autocomplete(parser)
@@ -213,10 +226,10 @@ def get_parser():
 
 def add_model_arguments(parser):
     parser.add_argument(\"--model\", \"-m\", default=\"vgg11\",
-             # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith(\"__\") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
-             help=\"model architecture\")
+                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith(\"__\") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
+                        help=\"model architecture\")
     parser.add_argument(\"--pretrained\", action=\"store_true\",
-             help=\"use ImageNet pretrained weights\")
+                        help=\"use ImageNet pretrained weights\")
     parser.add_argument(\"--load\", type=str, default=None, help=\"weights to load\")
 
     parser.add_argument(\"--checkpoint_every\", type=int, default=1, help=\"how frequently to save checkpoints\")
@@ -227,8 +240,10 @@ def add_model_arguments(parser):
 def add_dataset_arguments(parser):
     group = parser.add_mutually_exclusive_group(required=True)
     group.add_argument(\"--tcga\", action=\"store_const\", const=\"TCGA\", dest=\"dataset\", help=\"TCGA Breast Cancer\")
-    group.add_argument(\"--camelyon\", action=\"store_const\", const=\"Camelyon16\", dest=\"dataset\", default=False, help=\"Camelyon 16\")
-    group.add_argument(\"--spatial\", action=\"store_const\", const=\"Spatial\", dest=\"dataset\", default=False, help=\"Spatial\")
+    group.add_argument(\"--camelyon\", action=\"store_const\", const=\"Camelyon16\", dest=\"dataset\", default=False,
+                       help=\"Camelyon 16\")
+    group.add_argument(\"--spatial\", action=\"store_const\", const=\"Spatial\", dest=\"dataset\", default=False,
+                       help=\"Spatial\")
 
 
 def add_window_arguments(parser):
@@ -260,19 +275,26 @@ def add_training_arguments(parser):
 
     parser.add_argument(\"--workers\", type=int, default=4, help=\"number of workers for dataloader\")
 
+
 def add_gene_filter_arguments(parser):
     group = parser.add_mutually_exclusive_group()
-    group.add_argument(\"--gene_filter\", dest=\"gene_filter\", choices=[\"none\", \"high\", \"tumor\"], default=\"tumor\", help=\"special gene filters\")
+    group.add_argument(\"--gene_filter\", dest=\"gene_filter\", choices=[\"none\", \"high\", \"tumor\"], default=\"tumor\",
+                       help=\"special gene filters\")
     group.add_argument(\"--gene_list\", dest=\"gene_filter\", nargs=\"+\", type=str, help=\"specify list of genes to look at\")
-    group.add_argument(\"--gene_n\", dest=\"gene_filter\", type=int, help=\"specify number of genes to look at (top mean expressed)\")
+    group.add_argument(\"--gene_n\", dest=\"gene_filter\", type=int,
+                       help=\"specify number of genes to look at (top mean expressed)\")
 
 
 def add_normalization_arguments(parser):
     group = parser.add_mutually_exclusive_group()
-    group.add_argument(\"--norm\", action=\"store_const\", const=\"norm\", dest=\"norm\", default=None, help=\"normalize gene counts per spot\")
-    group.add_argument(\"--normfilter\", action=\"store_const\", const=\"normfilter\", dest=\"norm\", default=None, help=\"normalize gene counts per spot (filtered only)\")
-    group.add_argument(\"--normpat\", action=\"store_const\", const=\"normpat\", dest=\"norm\", default=None, help=\"normalize by median in patient\")
-    group.add_argument(\"--normsec\", action=\"store_const\", const=\"normsec\", dest=\"norm\", default=None, help=\"normalize by median in section\")
+    group.add_argument(\"--norm\", action=\"store_const\", const=\"norm\", dest=\"norm\", default=None,
+                       help=\"normalize gene counts per spot\")
+    group.add_argument(\"--normfilter\", action=\"store_const\", const=\"normfilter\", dest=\"norm\", default=None,
+                       help=\"normalize gene counts per spot (filtered only)\")
+    group.add_argument(\"--normpat\", action=\"store_const\", const=\"normpat\", dest=\"norm\", default=None,
+                       help=\"normalize by median in patient\")
+    group.add_argument(\"--normsec\", action=\"store_const\", const=\"normsec\", dest=\"norm\", default=None,
+                       help=\"normalize by median in section\")
 
 
 def add_logging_arguments(parser):
@@ -284,7 +306,7 @@ def add_logging_arguments(parser):
         return numeric_level
 
     parser.add_argument(\"--loglevel\", \"-l\", type=loglevel,
-                                     default=DEBUG, help=\"logging level\")
+                        default=DEBUG, help=\"logging level\")
     parser.add_argument(\"--logfile\", type=str,
-                                     default=None,
-                                     help=\"file to store logs\")
+                        default=None,
+                        help=\"file to store logs\")
diff --git a/tox.ini b/tox.ini
index 946ab45..7cb521b 100644
--- a/tox.ini
+++ b/tox.ini
@@ -1,5 +1,5 @@
 [tox]
-envlist = py36
+envlist = py37
 
 [testenv]
 deps ="""

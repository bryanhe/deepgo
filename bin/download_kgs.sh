#!/bin/bash

# See https://www.u-go.net/gamerecords/ for a human-readable page.

root="https://dl.u-go.net/gamerecords/"
files="
KGS-2018_11-19-1879-.tar.bz2
KGS-2018_10-19-1209-.tar.bz2
KGS-2018_09-19-1587-.tar.bz2
KGS-2018_08-19-1447-.tar.bz2
KGS-2018_07-19-949-.tar.bz2
KGS-2018_06-19-1002-.tar.bz2
KGS-2018_05-19-1590-.tar.bz2
KGS-2018_04-19-1612-.tar.bz2
KGS-2018_03-19-833-.tar.bz2
KGS-2018_02-19-1167-.tar.bz2
KGS-2018_01-19-1526-.tar.bz2
KGS-2017_12-19-1488-.tar.bz2
KGS-2017_11-19-945-.tar.bz2
KGS-2017_10-19-1351-.tar.bz2
KGS-2017_09-19-1353-.tar.bz2
KGS-2017_08-19-2205-.tar.bz2
KGS-2017_07-19-1191-.tar.bz2
KGS-2017_06-19-910-.tar.bz2
KGS-2017_05-19-847-.tar.bz2
KGS-2017_04-19-913-.tar.bz2
KGS-2017_03-19-717-.tar.bz2
KGS-2017_02-19-525-.tar.bz2
KGS-2017_01-19-733-.tar.bz2
KGS-2016_12-19-1208-.tar.bz2
KGS-2016_11-19-980-.tar.bz2
KGS-2016_10-19-925-.tar.bz2
KGS-2016_09-19-1170-.tar.bz2
KGS-2016_08-19-1374-.tar.bz2
KGS-2016_07-19-1432-.tar.bz2
KGS-2016_06-19-1540-.tar.bz2
KGS-2016_05-19-1011-.tar.bz2
KGS-2016_04-19-1081-.tar.bz2
KGS-2016_03-19-895-.tar.bz2
KGS-2016_02-19-577-.tar.bz2
KGS-2016_01-19-756-.tar.bz2
KGS-2015-19-8133-.tar.bz2
KGS-2014-19-13029-.tar.bz2
KGS-2013-19-13783-.tar.bz2
KGS-2012-19-13665-.tar.bz2
KGS-2011-19-19099-.tar.bz2
KGS-2010-19-17536-.tar.bz2
KGS-2009-19-18837-.tar.bz2
KGS-2008-19-14002-.tar.bz2
KGS-2007-19-11644-.tar.bz2
KGS-2006-19-10388-.tar.bz2
KGS-2005-19-13941-.tar.bz2
KGS-2004-19-12106-.tar.bz2
KGS-2003-19-7582-.tar.bz2
KGS-2002-19-3646-.tar.bz2
KGS-2001-19-2298-.tar.bz2"

mkdir -p data/kgs
cd data/kgs
for f in ${files}
do
  wget ${root}/${f} -O ${f}
  tar -xjf ${f}
done

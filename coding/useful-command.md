# useful command

Binary files, convenient command

## General

### check disk space

du -h --max-depth=1

### display file changes in real time

tail -f files.log

### view log files

less files.log tl files.log lnav files.log

### change directory easily

pushd

popd dirs

zi

### check port status

netstat -ano

### find public network ip

curl ifconfig.me

### find files with specific subfix

```bash
find / -type f -name "*.sh"
```

### check all disk space

```bash
lfs df -h -p sharefs.alipool /sharefs/alicpt
```

### hard link

ln -s /sharefs/your/path/to/directory /afs/your/path/to/directory

## Slurm operation

### ls column by index

```bash
ls -1 submit_run_*.sh | sort -V > do.sh
```

### add sbatch before submit\_\*.sh

```bash
sed -i 's/\([^ ]*\)/sbatch \1/g' do.sh
```

### find how many works are running

```bash
squeue -u wangyiming25 | grep ' R ' | wc -l
squeue -u wangyiming25 | grep ' PD ' | wc -l
```

### cancel homework from one id to another id

```bash
for job_id in {244328..244422}
do
   scancel $job_id
done
```

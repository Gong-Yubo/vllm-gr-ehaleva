# Development Utils Manual

This document introduces a set of development utilities to automate hardware allocation and streamline the local development workflow.

## Utility Specifications

- [Resource Helper](#resource-helper)
- [Pre-commit tester](#pre-commit-tester)
- [GitHub Actions tester](#github-actions-tester)
- [Export branch](#export-branch)

### Resource Helper

A collection of functions to manage local GPU pool resource sharing.  
Resource sharing assumes all users of the local server use this procedure.

- **get_gpu** `[alloc]` - Search for a free GPU.  
  `alloc` - Maximum allocated RAM in MB (default: 4).

**Returns:** GPU ID (blocks until a GPU is free).

```bash
source ./tools/resource.sh
id=$(get_gpu 5) # Scans for a free GPU with at most 5 MB allocated RAM.
```

- **get_gpus** \<`num`\> [`alloc`] - get number of gpus  
    `num` - number of requested GPUs  
    `alloc` - most allocated RAM in MB (default 4)

Returns: comma separated list of available gpu's , search  until found

```bash
source ./tools/resource.sh
ids=$(get_gpus 3 4) # Scans for 3 free GPU's with at most 4 MB allocated RAM.
CUDA_VISIBLE_DEVICES=$ids ./run_some_tests.sh
```

- **lock_gpu** \<`id`\> - preserve  a gpu (by allocating some memory to it)  
    `id` - GPU id

Returns: pid of preserving memory process

``` bash
source ./tools/resource.sh
id=$(get_gpu 4)
pid=$(lock_gpu $id)  
```

- **lock gpus** \<`ids`\> - preserve number of gpus from the given comma separated list.  
`ids` - comma separated list of GPU id

Returns : Comma separated of pids of preserved memory processes

``` bash
source ./tools/resource.sh
# default permitted idle GPU allocation is 4MB
ids=$(get_gpus)
pids=$(lock_gpus $ids)  
```

- **unlock_gpu** \<pid\> - unlock preserved gpu
pid - process id of holding preserved memory process

```bash
source ./tools/resource.sh
id=$(get_gpu 4)
pid=$(lock_gpu $id)
CUDA_VISIBLE_DEVICES=$id do_something.sh
unlock_gpu $pid  #unlock gpu by killing its allocator process
```

- **unlock_gpus** <`pids`> - unlock preserved gpus  
`pids` - comma separated list of process id of holding preserved memory processes

```bash
source ./tools/resource.sh
ids=$(get_gpus 3)
pids=$(lock_gpus $ids)
CUDA_VISIBLE_DEVICES=$ids do_something
unlock_gpus $pids # unlock gpus by killing its allocator processes
```

### Pre-commit tester

Run full pre-commit test including unit tests. It uses get_gpu, lock_gpu and unlock_gpu  
This script assumes:

- pre-commit package is installed, if not please install with:

    ``` bash
    pip install pre-commit
    ```

    Refer to [pre-commit](https://github.com/pre-commit/pre-commit) for tool description and installation instruction

- vllm-gr is installed, see [quickstart](../quickstart.md).

Run test with:

```bash
./tools/run_test.sh
```

### Github Actions tester

Execute a full workflows execution. Its create an empty environment via docker and perform all workflows define in .github/workflows.
This script assume act is installed on system. Please refer to   [nektos act](https://github.com/nektos/act) for tool information and perquisites and check  [act installation](https://nektosact.com/installation/) for installation details.

>[!NOTE]
HuggingFace token (HF_TOKEN) need to be supplied because workflow include OpenOneRec dataset downloading.
Refer to [hugging-face](https://huggingface.co/settings/tokens) for instruction to get a token,

```bash
HF_TOKEN=<xxxx>   ./tools/run_act.sh
```

### Export branch

A script to clean up the branch history before a Pull Request
Automatically Squashes all commits into a single clean commit.

**usage:**

```bash
git pull upstream main                 # update from main branch
git commit -a                          # commit merges, conflicts etc.
./tools/export_changes.sh              # create a new branch with 1 commit
git commit -a                          # commit the changes
git push origin HEAD:<your pr>         # push to PR branch
```

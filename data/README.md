# NSL-KDD Dataset

## Overview

The NSL-KDD dataset is an improved version of the KDD'99 dataset, which is widely used for network intrusion detection research. It addresses some of the inherent problems of the KDD'99 dataset.

## Dataset Statistics

- **Training Set**: KDDTrain+.txt (125,973 records)
- **Test Set**: KDDTest+.txt (22,544 records)
- **Features**: 41 + 1 label + 1 difficulty score = 43 columns
- **Attack Types**: 4 main categories + Normal
  - DoS (Denial of Service)
  - Probe (Surveillance and Probing)
  - R2L (Remote to Local)
  - U2R (User to Root)

## Features

### Basic Features (9)
1. duration
2. protocol_type (tcp, udp, icmp)
3. service (http, ftp, smtp, etc.)
4. flag (connection status)
5. src_bytes
6. dst_bytes
7. land
8. wrong_fragment
9. urgent

### Content Features (13)
10. hot
11. num_failed_logins
12. logged_in
13. num_compromised
14. root_shell
15. su_attempted
16. num_root
17. num_file_creations
18. num_shells
19. num_access_files
20. num_outbound_cmds
21. is_host_login
22. is_guest_login

### Traffic Features - Time-based (9)
23. count
24. srv_count
25. serror_rate
26. srv_serror_rate
27. rerror_rate
28. srv_rerror_rate
29. same_srv_rate
30. diff_srv_rate
31. srv_diff_host_rate

### Traffic Features - Host-based (10)
32. dst_host_count
33. dst_host_srv_count
34. dst_host_same_srv_rate
35. dst_host_diff_srv_rate
36. dst_host_same_src_port_rate
37. dst_host_srv_diff_host_rate
38. dst_host_serror_rate
39. dst_host_srv_serror_rate
40. dst_host_rerror_rate
41. dst_host_srv_rerror_rate

### Labels
42. attack_type (normal, or specific attack name)
43. difficulty_level (integer score)

## Download Instructions

```bash
# Run the download script
python data/download_dataset.py
```

The script will download NSL-KDD dataset files to the `data/` directory.

## Alternative Download

If the automated download fails, manually download from:
- https://www.unb.ca/cic/datasets/nsl.html
- Or Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd

Place the following files in the `data/` directory:
- KDDTrain+.txt
- KDDTest+.txt

## Citation

```
M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, 
"A Detailed Analysis of the KDD CUP 99 Data Set," 
Submitted to Second IEEE Symposium on Computational Intelligence 
for Security and Defense Applications (CISDA), 2009.
```

## License

NSL-KDD dataset is publicly available for research purposes.

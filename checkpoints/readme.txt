================================================================
Documentation
================================================================ 
1. Folder Naming Convention
================================================================
XXXX is a 4-digit ID.
Digits 1-2 → kernel size of the 1-st convolution branch in GRAttention.
Digits 3-4 → kernel size of the 2-nd convolution branch.
Examples
33  → [3×3]
3377  → [3×3, 7×7]
5577  → [5×5, 7×7]
113377  → [1×1, 3×3, 7×7]
================================================================ 
2. File Description
================================================================ 
train.log
 Raw training log (plain text, space-separated).
 Column order: epoch batch loss lr timestamp [other_metrics]
train.csv
 Structured table exported from train.log.
 Header: epoch,batch,loss,lr,timestamp,...
================================================================ 
3. Comparing Different Kernels
================================================================ 
Load several csv files, add a kernel column.
Group by kernel and plot validation loss/accuracy.
Pick the kernel set with the best trade-off between speed and accuracy.
================================================================
4. Copyright & Maintenance
================================================================
This README.txt was manually written.
For questions or suggestions please contact:

zsliu0304@gmail.com

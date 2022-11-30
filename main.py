from basic import Basic


# 创建类
classification = Basic()
# 定义使用的method
# 只修改method即可
method = 'RandomForest'
pred, pred_proba = classification.predict(method)
# 输出正确率, 混淆矩阵和错误编号
error_index = classification.print_results(pred)
# 绘制ROC曲线并计算AUC
classification.plot_roc(pred_proba, method)
# 保存错误编号
classification.write_error_index(error_index)

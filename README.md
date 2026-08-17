# Plasma_Etch_OES_Virtual_Metrology_NPU

| 파일 | 안에 들어있는 것 | 앞으로 중요도 |
| --- | --- | --- |
| `dataset_2024_08_05.npz` | 2024-08-05의 5개 wafer만 만든 초기 테스트 데이터셋 | 낮음 |
| **`full_dataset_raw.npz`** | 전체 75 wafer의 `X(75,100,128,1)`, target `y`, lot, wafer 번호 등 | **매우 중요** |
| `full_dataset_qc.csv` | 75개 wafer의 cycle, offset, correlation, PASS/FAIL 전처리 QC 결과 | 기록용 중요 |
| `usable_experiments.csv` | 실제 사용할 수 있는 75개 wafer의 날짜/Lot/Wafer 목록 | 기록용 |
| `tiny_cnn_lolo_result.npz` | 처음 실패했던 Direct CNN의 target/prediction 결과 | 분석/트러블슈팅용 |
| `oes_residual_result.npz` | Wafer baseline + OES Ridge residual의 예측 결과 | 비교 baseline |
| `residual_tiny_cnn_result.npz` | 성공한 Residual Tiny CNN의 LOLO 예측 결과 | AI 결과 기록 |
| `multiseed_predictions.npz` | 5개 seed 각각의 CNN 예측값 | 안정성 검증 기록 |
| `multiseed_summary.csv` | seed별 MAE/RMSE/R²/개선 Lot 수 | 발표/보고서용 |
| `multiseed_lot_results.csv` | 각 seed × 각 Lot의 세부 성능 | 상세 분석용 |
| `int8_ptq_result.npz` | FP32 예측과 INT8 예측을 비교한 결과 | **INT8 검증 증거** |
| **`deployment_fp32_model.pt`** | 75개 전체로 학습한 최종 PyTorch CNN의 FP32 `state_dict` | **매우 중요** |
| **`deployment_int8_model.npz`** | FPGA에 사용할 INT8 Weight, INT32 Bias, activation/weight scale, baseline 계수 등 | **RTL 핵심 파일** |
| **`fixed_requant_params.npz`** | Conv1/Conv2/FC1의 `M`, `Shift=24`, output scale | **RTL 핵심 파일** |
| **`fixed_golden_result.npz`** | 전체 75개에 대해 FPGA 방식 fixed-point inference를 돌린 최종 Golden 결과 | **RTL 검증 핵심** |
| **`rtl_golden_sample0.npz`** | 특정 wafer 1개의 입력부터 각 Layer 출력까지 전부 저장 | **RTL 디버깅 최핵심** |

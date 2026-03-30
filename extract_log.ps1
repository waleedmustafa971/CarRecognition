$lines = Get-Content 'C:\Users\Administrator\Desktop\car_recognition\logs\service_stderr.log'
$filtered = $lines | Where-Object { $_ -match 'NEW REQUEST' }
$filtered | Select-Object -Last 20 | ForEach-Object { $_.Trim() }

 = New-ScheduledTaskAction -Execute "docker" -Argument "run --rm -v D:\Crypto-Trading-Bot:/app training_container"
 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am
Register-ScheduledTask -TaskName "CryptoBotRetraining" -Action  -Trigger 

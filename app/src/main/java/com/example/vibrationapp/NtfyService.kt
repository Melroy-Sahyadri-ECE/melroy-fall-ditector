package com.example.vibrationapp

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.Ringtone
import android.media.RingtoneManager
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.core.app.NotificationCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import java.util.concurrent.TimeUnit

class NtfyService : Service() {

    private var vibrator: Vibrator? = null
    private var ringtone: Ringtone? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private var eventSource: EventSource? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    companion object {
        const val ACTION_STOP_ALARM = "com.example.vibrationapp.STOP_ALARM"
        const val CHANNEL_ID = "NtfyServiceChannel"
    }

    override fun onCreate() {
        super.onCreate()
        
        // Initialize WakeLock
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "VibrationApp::NtfyWakeLock")
        wakeLock?.acquire(10 * 60 * 1000L /*10 minutes max just in case, but usually we just hold it while running or we can hold it forever if 24/7 is needed*/)
        wakeLock?.acquire()

        // Initialize Vibrator
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // Initialize Ringtone
        val defaultRingtoneUri: Uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
        ringtone = RingtoneManager.getRingtone(applicationContext, defaultRingtoneUri)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ringtone?.isLooping = true
        }

        startNtfyListener()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()

        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, NtfyService::class.java).apply {
            action = ACTION_STOP_ALARM
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Fall Detection Listener")
            .setContentText("Listening for fall events 24/7...")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .addAction(0, "Stop Alarm", stopPendingIntent)
            .build()

        startForeground(1, notification)

        if (intent?.action == ACTION_STOP_ALARM) {
            stopVibration()
        }

        return START_STICKY
    }

    private fun startNtfyListener() {
        val request = Request.Builder()
            .url("https://ntfy.sh/melroy-fall-detector/sse")
            .build()

        val listener = object : EventSourceListener() {
            override fun onOpen(eventSource: EventSource, response: Response) {
                // Connected
            }

            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data.contains("\"event\":\"message\"") || type == "message") {
                    startVibration()
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                // Try to reconnect after 3 seconds
                eventSource.cancel()
                Thread.sleep(3000)
                startNtfyListener()
            }
        }

        eventSource = EventSources.createFactory(client).newEventSource(request, listener)
    }

    private fun startVibration() {
        ringtone?.play()
        vibrator?.let {
            if (it.hasVibrator()) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val vibrationEffect = VibrationEffect.createWaveform(longArrayOf(0, 1000), 0)
                    it.vibrate(vibrationEffect)
                } else {
                    @Suppress("DEPRECATION")
                    it.vibrate(longArrayOf(0, 1000), 0)
                }
            }
        }
    }

    private fun stopVibration() {
        ringtone?.stop()
        vibrator?.cancel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                CHANNEL_ID,
                "Ntfy Background Service Channel",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(serviceChannel)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        eventSource?.cancel()
        stopVibration()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }
}

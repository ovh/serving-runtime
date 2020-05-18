package com.ovh.mls.serving.runtime.timeseries;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;

import static java.time.temporal.ChronoUnit.*;

/**
 * Encoders / Decoders for (LocalDateTime / Long)
 */
public enum LocalDateTimeEncoder {
    // Numeric representation of the year
    YEAR,
    // Numeric representation of the month
    MONTH,
    // Numeric representation of the day of the month
    DAYOFMONTH,
    // Numeric representation of the day of the week
    DAYOFWEEK,
    // Numeric representation of the hour of the day
    HOUROFDAY,
    // Numeric representation of the minute of the hour
    MINUTEOFHOUR,
    // Timestamp UTC in years
    TS_Y,
    // Timestamp UTC in months
    TS_MO,
    // Timestamp UTC in days
    TS_D,
    // Timestamp UTC in hours
    TS_H,
    // Timestamp UTC in minutes
    TS_MI,
    // Timestamp UTC in seconds
    TS_S,
    // Timestamp UTC in milliseconds
    TS_MS,
    // Timestamp UTC in microseconds
    TS_US,
    // Timestamp UTC in nanoseconds
    TS_NS;

    private static final ZoneOffset REF_ZONE = ZoneOffset.UTC;
    private static final LocalDateTime REF_DATETIME = LocalDateTime.ofInstant(Instant.ofEpochMilli(0), REF_ZONE);

    public static LocalDateTimeEncoder fromString(String name) {
        return valueOf(name.toUpperCase());
    }

    public LocalDateTime decode(long timestamp) {
        switch(this) {
            case TS_Y: return REF_DATETIME.plusYears(timestamp);
            case TS_MO: return REF_DATETIME.plusMonths(timestamp);
            case TS_D: return REF_DATETIME.plusDays(timestamp);
            case TS_H: return REF_DATETIME.plusHours(timestamp);
            case TS_MI: return REF_DATETIME.plusMinutes(timestamp);
            case TS_S: return REF_DATETIME.plusSeconds(timestamp);
            case TS_MS: return LocalDateTime.ofInstant(Instant.ofEpochMilli(timestamp), REF_ZONE);
            case TS_US: return LocalDateTime.ofInstant(
                Instant.ofEpochSecond(
                    TimeUnit.MICROSECONDS.toSeconds(timestamp),
                    TimeUnit.MICROSECONDS.toNanos(
                        Math.floorMod(timestamp, TimeUnit.SECONDS.toMicros(1))
                    )
                ), REF_ZONE);
            case TS_NS: return LocalDateTime.ofInstant(Instant.ofEpochSecond(0L, timestamp), REF_ZONE);
            default:
                throw new AssertionError(
                    String.format("Impossible to construct a date from a %s only transformation", this));
        }
    }

    public Long encode(LocalDateTime dateTime) {
        switch(this) {
            case YEAR: return (long) dateTime.getYear();
            case MONTH: return (long) dateTime.getMonthValue();
            case DAYOFMONTH: return (long) dateTime.getDayOfMonth();
            case DAYOFWEEK: return (long) dateTime.getDayOfWeek().getValue();
            case HOUROFDAY: return (long) dateTime.getHour();
            case MINUTEOFHOUR: return (long) dateTime.getMinute();

            case TS_Y: return YEARS.between(REF_DATETIME, dateTime);
            case TS_MO: return MONTHS.between(REF_DATETIME, dateTime);
            case TS_D: return WEEKS.between(REF_DATETIME, dateTime);
            case TS_H: return HOURS.between(REF_DATETIME, dateTime);
            case TS_MI: return MINUTES.between(REF_DATETIME, dateTime);

            case TS_S: return SECONDS.between(REF_DATETIME, dateTime);
            case TS_MS: return MILLIS.between(REF_DATETIME, dateTime);
            case TS_US: return MICROS.between(REF_DATETIME, dateTime);
            case TS_NS: return NANOS.between(REF_DATETIME, dateTime);
        }
        throw new AssertionError("Unknown operation : " + this);
    }

}

